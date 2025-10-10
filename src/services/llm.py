import os
import time
from typing import Dict, List, Generator, Optional
from openai import OpenAI
import re
from src.services.citations import (
    find_and_verify_citations,
    format_references_block,
    format_fallback_block,
)
try:  # v2 citations optional import
    from src.services.citations_v2.pipeline import gather_verified_citations as citations_v2_gather
except Exception:  # pragma: no cover
    citations_v2_gather = None


SYSTEM_PROMPT = """
You are WellnessGPT, an advanced AI wellness companion powered by an agentic, multi-expert framework. 
You proactively guide users through personalized health conversations, combining clinical reasoning, 
lifestyle coaching, genomic insights, and emotional intelligence.

You coordinate an internal team of expert agents:
- Dr. Core - clinical reasoning & diagnostics
- Coach Kai - lifestyle & habit optimization
- Analyst Ava - genomic and data interpretation
- Empath Elle - emotional tone & trust-building

Your role is to integrate agent outputs into one seamless, supportive reply.

---

Core Functionality
- Act as both a wellness coach and proactive health companion.
- Keep replies concise (~3 sentences), in plain English, with actionable advice on diet, exercise, sleep, stress, and mental health.
- Interpret uploaded genomic data (e.g., specific rsIDs) and embed findings into diagnostics and recommendations.
- Use agentic routing: call on agents internally, cite them when relevant.
- Adapt follow-up questions dynamically based on symptoms, behaviors, or red flags.
- Track conversation memory to ensure continuity across sessions.
---

Interpretability & Transparency
- For key questions, generate a short rationale in plain English, optionally surfaced as:
  "Why this? This helps assess your [goal] based on [logic]."
- At session end, generate a self-audit summary explaining why each major question was asked.

---

Dynamic Behavior
- Detect concerning symptom clusters (e.g., fatigue + weight loss) and escalate with follow-up screening questions.
- If multiple red flags persist, recommend clinical action (e.g., "Please consider seeing your doctor for blood work.").
- Ensure genomic insights are contextual, accurate, and smoothly integrated.

---

Session Wrap-Up
When the user signals they are finished, produce a structured summary including:
1. Top concerns & risk patterns
2. Key recommendations by agent (Dr. Core, Coach Kai, Analyst Ava, Empath Elle)
3. Audit trail: why each major question was asked
4. Supporting references or mock citations (real ones if tool access is enabled)

---

Engagement & Tone
- Empathetic, concise, and encouraging — adjust tone to match user mood.
- Use creative metaphors or motivational nudges when helpful.
- Non-judgmental at all times.
- End each reply with a proactive, clarifying follow-up to keep conversation flowing.

---

Expert Finder Capability (Phase 1)
If the user asks to find a healthcare professional (e.g., "find a dermatologist near me"), a separate UI card will appear that performs the location-based search. In your reply:
1. Acknowledge the specialty and that a helper panel will assist with nearby providers.
2. Offer 1–2 brief preparation tips (e.g., symptoms to note, records, questions to ask).
3. Provide general wellness or triage considerations (red flags that warrant urgent care if any).
4. Do NOT say you cannot search; the UI handles it. Keep this acknowledgement to 1 short sentence before continuing normal guidance.
"""


def _now() -> str:
    return time.ctime(time.time())

def _ensure_question(text: str) -> str:
    """
    Safety net: if the model forgot to end with a question, append a short one.
    """
    t = (text or "").strip()
    if "?" in t: return t
    return t + "\n\nWhat would you like to focus on next?"

class LlmBroker:
    """
    Minimal broker around OpenAI Chat Completions.
    - Keeps per-session message history in-memory (swap for DB if needed).
    - Provides sync and streaming replies.
    - Enforces conversational follow-ups.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt: str = SYSTEM_PROMPT,
        default_temperature: float = 0.6,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("api_key") or ""
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model
        self.system_prompt = system_prompt
        self.default_temperature = default_temperature

        # session_id -> list[{"role": "user"|"assistant", "content": str, "time": str}]
        self.histories: Dict[str, List[Dict]] = {}

        self.always_cite = True          # <— force citations for EVERY reply (testing)
        self.citation_verify_min_score = 0  # <— looser filter during testing
        # Memory condensation settings
        self.memory_condense_threshold = 22   # when history messages exceed this, condense
        self.memory_keep_tail = 8             # keep last N detailed turns
        self._last_condense_len = {}

    def _compact_query(self, *texts, max_terms=12):
        bag, seen = [], set()
        for t in texts:
            for w in re.findall(r"[A-Za-z]{4,}", (t or "")):
                lw = w.lower()
                if lw not in seen:
                    bag.append(lw); seen.add(lw)
                if len(bag) >= max_terms:
                    break
        return " ".join(bag)

    def _attach_citations(self, user_input: str, answer_text: str) -> str:
        """Attach citations (preferring v2 pipeline) unless already present."""
        if "References:" in answer_text:
            return answer_text
        # Strip any accidental model-added 'References' preface during generation
        cleaned = re.split(r"\n\s*References\s*:|\n\s*References\s*\n", answer_text, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()
        max_results = 3
        # Try v2 first unless disabled
        if not os.getenv("CITATIONS_V2_DISABLE") and citations_v2_gather:
            try:
                v2_items = citations_v2_gather(user_input, cleaned, max_results=max_results)
                if v2_items:
                    enriched = []
                    for c in v2_items:
                        title = c.get("title", "Reference")
                        meta_bits = []
                        if c.get("evidence_level"):
                            meta_bits.append(c["evidence_level"].replace("_", " "))
                        if c.get("year"):
                            meta_bits.append(str(c["year"]))
                        if meta_bits:
                            title = f"{title} ({', '.join(meta_bits)})"
                        enriched.append({"source": c.get("source"), "title": title, "url": c.get("url")})
                    block = format_references_block(enriched)
                    return cleaned + block
            except Exception:
                pass
        # Legacy fallback
        try:
            verified = find_and_verify_citations(user_input, cleaned, max_results=max_results)
            if verified:
                return cleaned + format_references_block(verified)
            return cleaned + format_fallback_block()
        except Exception:
            return cleaned

    # ---------- memory condensation ----------
    def maybe_condense_history(self, session_id: str, force: bool = False) -> bool:
        """
        If history is long, condense older portion into a single summary message to keep context lean.
        Returns True if condensation performed.
        """
        hist = self.get_history(session_id)
        length = len(hist)
        if not force:
            if length < self.memory_condense_threshold:
                return False
            # Avoid re-condensing too often
            last_len = self._last_condense_len.get(session_id, 0)
            if length - last_len < 6:  # need at least 6 new turns before another condense
                return False
        # Identify portion to condense (exclude existing memory summary markers & keep tail)
        tail = hist[-self.memory_keep_tail:]
        head = hist[:-self.memory_keep_tail]
        # If head already a single memory summary skip
        if len(head) <= 1 and head and head[0]["content"].startswith("[Memory Summary]") and not force:
            return False
        summary_text = self._generate_condensed_summary(head, session_id)
        # Rebuild history: memory summary + tail
        new_hist = [{"role": "assistant", "content": summary_text, "time": _now()}]
        new_hist.extend(tail)
        self.histories[session_id] = new_hist
        self._last_condense_len[session_id] = len(new_hist)
        return True

    def _generate_condensed_summary(self, messages: List[Dict], session_id: str) -> str:
        """Use model (if available) or heuristic to compress earlier turns."""
        if not messages:
            return "[Memory Summary] (empty)"
        # Heuristic dev fallback
        if not self.client:
            bullets = []
            for m in messages:
                if m["role"] == "user":
                    bullets.append(f"- User: {m['content'][:140]}")
                elif m["role"] == "assistant" and not m["content"].startswith("[Memory Summary]"):
                    bullets.append(f"- Assistant: {m['content'][:140]}")
                if len(bullets) >= 10:
                    break
            return "[Memory Summary]\n" + "\n".join(bullets) + "\n[End Summary]"
        # Use model summarization
        convo = []
        for m in messages[-40:]:
            if m["content"].startswith("[Memory Summary]"):
                continue
            role = "User" if m["role"] == "user" else "Assistant"
            convo.append(f"{role}: {m['content']}")
        convo_text = "\n".join(convo)
        prompt = f"""Condense the earlier portion of this conversation into ~6 bullet points capturing:
        - User main concerns / goals
        - Any symptoms / risk flags
        - Recommendations already given
        - Outstanding follow-ups
        Keep it concise; no citations. Conversation:\n{convo_text}\n"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "You produce concise memory summaries."},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = completion.choices[0].message.content or "(summary unavailable)"
        except Exception:
            txt = "(summary unavailable)"
        return "[Memory Summary]\n" + txt.strip() + "\n[End Summary]"

    # ---------- simple tool planning (keyword heuristic) ----------
    def _planned_tools(self, user_input: str) -> List[str]:
        low = user_input.lower()
        planned = []
        if any(k in low for k in ("source", "citation", "reference", "evidence")):
            planned.append("citations")
        if any(k in low for k in ("summarize", "summary", "recap")):
            planned.append("memory_summarize")
        if any(k in low for k in ("email me", "send me", "mail this")):
            planned.append("offer_email_summary")
        return planned

    def _should_cite(self, user_input: str, reply_text: str, force: bool = False) -> bool:
        if getattr(self, "always_cite", False): return True
        if force: return True
        return len(reply_text) >= 120


    # ---------- history ----------
    def get_history(self, session_id: str) -> List[Dict]:
        return self.histories.setdefault(session_id, [])

    def history_size(self, session_id: str) -> int:
            return len(self.get_history(session_id))

    def add_user(self, session_id: str, content: str) -> None:
        self.get_history(session_id).append({"role": "user", "content": content, "time": _now()})

    def add_assistant(self, session_id: str, content: str) -> None:
        self.get_history(session_id).append({"role": "assistant", "content": content, "time": _now()})

    def clear(self, session_id: str) -> None:
        self.histories[session_id] = []

    # ---------- message assembly ----------
    def _messages(self, session_id: str) -> List[Dict]:
        history = self.get_history(session_id)
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend({"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant"))
        return msgs
    
    # ---------- sync reply ----------
    def reply_sync(
        self,
        session_id: str,
        user_input: str,
        temperature: Optional[float] = None,
        force_citations: bool = False,
        intent_context: Optional[str] = None,
    ) -> str:
        """Synchronous reply.

        intent_context: optional extra system message inserted AFTER the primary system prompt
          (not persisted in history) to nudge behavior for detected intents (e.g., expert finder).
        """
        self.add_user(session_id, user_input)
        # Maybe condense before generating reply to keep context small
        self.maybe_condense_history(session_id)
        planned = self._planned_tools(user_input)

        if not self.client:  # dev fallback without API key
            prefix = ""
            if intent_context:
                # Keep short; intent_context already descriptive
                prefix = "(Dev mode) Expert Finder active. "
            reply = f"{prefix}You said: {user_input}"
            reply = _ensure_question(reply)
            # Always attach citations in dev as well
            reply = self._attach_citations(user_input, reply)
            self.add_assistant(session_id, reply)
            return reply

        msgs = self._messages(session_id)
        if intent_context:
            # Insert right after main system prompt (index 0) so it has high priority
            msgs.insert(1, {"role": "system", "content": intent_context})

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self._temp(temperature),
            messages=msgs,
        )
        reply = completion.choices[0].message.content or "Sorry, I couldn't generate a response."
        reply = _ensure_question(reply)
        # Inline lightweight tool outputs (non-invasive)
        if "memory_summarize" in planned:
            # Generate a quick tail summary (not altering history directly here)
            try:
                mem = self._generate_condensed_summary(self.get_history(session_id)[:-1], session_id)
                reply = reply + "\n\n(Quick recap just for you)\n" + "\n".join(mem.splitlines()[:8])
            except Exception:
                pass
        if "offer_email_summary" in planned:
            reply += "\n\nIf you'd like a copy by email, enter your address in the summary card and click 'Email Summary'."

        # --- FORCE citations on every reply ---
        reply = self._attach_citations(user_input, reply)

        self.add_assistant(session_id, reply)
        return reply

    # ---------- streaming reply (yields tokens) ----------
    def stream_reply(
        self,
        session_id: str,
        user_input: str,
        temperature: Optional[float] = None,
        force_citations: bool = False,
        intent_context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        self.add_user(session_id, user_input)
        self.maybe_condense_history(session_id)
        planned = self._planned_tools(user_input)

        if not self.client:  # dev fallback without API key
            pre = "" if not intent_context else "(Dev mode) Expert Finder active. "
            base = f"{pre}You said: {user_input}. What should we tackle next?"
            final = _ensure_question(base)
            final = self._attach_citations(user_input, final)
            # Emit as a single token chunk for dev mode
            yield final
            self.add_assistant(session_id, final)
            return

        msgs = self._messages(session_id)
        if intent_context:
            msgs.insert(1, {"role": "system", "content": intent_context})

        acc = ""
        with self.client.chat.completions.stream(
            model=self.model,
            temperature=self._temp(temperature),
            messages=msgs,
        ) as stream:
            for event in stream:
                if event.type == "token":
                    token = event.token
                    acc += token
                    yield token
                elif event.type == "completed":
                    final_text = _ensure_question(acc)
                    if "offer_email_summary" in planned:
                        final_text += "\n\n(You can email this session summary using the summary card.)"
                    final_text = self._attach_citations(user_input, final_text)
                    self.add_assistant(session_id, final_text)
                    return

                elif event.type == "error":
                    err_text = f"[OpenAI error: {getattr(event, 'error', 'unknown')}]"
                    self.add_assistant(session_id, err_text)
                    yield err_text
                    return

    # ---------- config helpers ----------
    def set_system_prompt(self, text: str) -> None:
        self.system_prompt = text

    def _temp(self, t: Optional[float]) -> float:
        return float(self.default_temperature if t is None else t)
    
    # --- ADD THIS inside class LlmBroker in src/services/llm.py ---

    def session_summary(self, session_id: str) -> str:
        """
        Produce a structured wellness session summary (agentic, multi-expert).
        Returns Markdown text.
        """
        history = self.get_history(session_id)
        # If no API key, produce a simple local summary
        if not self.client:
            bullets = []
            for m in history[-10:]:
                if m["role"] == "user":
                    bullets.append(f"- {m['content']}")
            body = "\n".join(bullets) or "- No recent user messages."
            return (
                "# Wellness Summary (Dev Mode)\n\n"
                "## Top Concerns\n" + body + "\n\n"
                "## Recommendations by Agent\n"
                "- **Dr. Core**: Consider a baseline checkup if symptoms persist.\n"
                "- **Coach Kai**: 30 minutes of movement daily; consistent sleep.\n"
                "- **Analyst Ava**: Upload genomic data for rsID insights.\n"
                "- **Empath Elle**: You’re making progress—small steps compound.\n\n"
                "## Next Steps\n- Set a 1-week goal\n- Keep a brief symptom log\n- Check back next week\n"
            )

        # With API key: ask the model for a concise, structured summary
        # Build a compact transcript
        turns = []
        for m in history[-20:]:
            role = "User" if m["role"] == "user" else "Assistant"
            turns.append(f"{role}: {m['content']}")
        transcript = "\n".join(turns) or "No conversation yet."

        prompt = f"""
You are WellnessGPT compiling a structured session summary using your agentic, multi-expert framework.

Conversation (last ~20 turns):
{transcript}

Create a concise Markdown summary with these sections:
1) Top Concerns & Patterns (bullets)
2) Recommendations by Agent
   - Dr. Core (clinical)
   - Coach Kai (lifestyle)
   - Analyst Ava (genomic if any rsIDs mentioned; otherwise note pending)
   - Empath Elle (tone/coaching)
3) Concrete Next Steps (3–5 bullets)
4) One-line Follow-up Question (to keep the conversation going)

Be precise, plain English, ~150–250 words total.
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self._temp(0.5),
            messages=[
                {"role": "system", "content": "You are a helpful wellness summarizer that writes clear Markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        text = completion.choices[0].message.content or "Summary unavailable."
        return text.strip()
