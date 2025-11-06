import os
import time
import re
import traceback
import json
from typing import Dict, List, Generator, Optional

# small helper
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Optional Groq adapter (prefer if GROQ_API_KEY present)
try:
    from src.services.groq_client import GroqClient
except Exception:
    GroqClient = None

# Citations helpers (best-effort imports)
try:
    from src.services.citations import (
        find_and_verify_citations,
        format_references_block,
        format_fallback_block,
    )
except Exception:
    # Provide fallbacks so module import doesn't fail; functions will raise if used.
    def find_and_verify_citations(*args, **kwargs):
        return []

    def format_references_block(items):
        return "\n\nReferences:\n" + "\n".join([f"- {i.get('source','?')} — {i.get('title','') or ''}" for i in items])

    def format_fallback_block():
        return "\n\nReferences: (no verified sources available)"

# Optional v2 citations pipeline
try:
    from src.services.citations_v2.pipeline import gather_verified_citations as citations_v2_gather
except Exception:
    citations_v2_gather = None

# Survey support (optional)
try:
    from src.services.surveys import SurveyManager
except Exception:
    SurveyManager = None

SYSTEM_PROMPT = """
    You are WellnessGPT, an advanced AI wellness companion powered by an agentic, multi-expert framework. 
    You proactively guide users through personalized health conversations, combining clinical reasoning, 
    lifestyle coaching, genomic insights, and emotional intelligence.

    You balance the following expert roles:
    - Clinical reasoning & diagnostics
    - Lifestyle & habit optimization
    - Genomic and data interpretation
    - Emotional tone & trust-building

    Your role is to integrate each output into one seamless, supportive reply.

    ---
"""


def _ensure_question(text: str) -> str:
    """
    Safety net: if the model forgot to end with a question, append a short one.
    """
    t = (text or "").strip()
    if "?" in t:
        return t
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
        model: str = "llama-3.1-8b-instant",
        system_prompt: str = SYSTEM_PROMPT,
        default_temperature: float = 0.6,
    ):
        # Prefer GROQ if configured (reads from env GROQ_API_KEY)
        groq_key = os.getenv("GROQ_API_KEY")
        groq_imported = GroqClient is not None
        if groq_key and groq_imported:
            try:
                self.client = GroqClient(api_key=groq_key)
                self.provider = "Groq"
            except Exception as e:
                print(f"[llm] Failed to init GroqClient: {e}")
                if os.getenv("AVATAR_DEBUG") == "1":
                    traceback.print_exc()
                self.client = None
                self.provider = "dev"
        else:
            # No Groq configured -> run in dev mode (no external LLM)
            self.client = None
            self.provider = "dev"
        # Diagnostic summary (mask keys)
        try:
            groq_present = bool(os.getenv("GROQ_API_KEY"))
            print(f"[llm] provider={self.provider} GROQ_KEY={'set' if groq_present else 'unset'} GroqClientImported={groq_imported}")
        except Exception:
            pass
        # Log chosen provider for debugging
        try:
            print(f"[llm] provider={self.provider}")
        except Exception:
            pass
        # Allow overriding the default model for Groq via env var GROQ_MODEL
        self.model = os.getenv("GROQ_MODEL") or model
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

        # Survey manager (loads surveys.json and tracks per-session survey state)
        try:
            self.survey_manager = SurveyManager()
        except Exception:
            self.survey_manager = None

    def _compact_query(self, *texts, max_terms=12):
        bag, seen = [], set()
        for t in texts:
            for w in re.findall(r"[A-Za-z]{4,}", (t or "")):
                lw = w.lower()
                if lw not in seen:
                    bag.append(lw)
                    seen.add(lw)
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

    # ---------- survey helpers ----------
    def _record_survey_response_if_expected(self, session_id: str, user_input: str) -> Optional[str]:
        """If the session is awaiting a survey response, record it and return an acknowledgement reply."""
        if not getattr(self, "survey_manager", None):
            return None
        st = self.survey_manager.get_state(session_id)
        if not st:
            return None
        waiting = st.get("waiting_for")
        if not waiting:
            # Nothing expected right now
            return None
        # Use ONLY LLM-based classification to decide whether to record the reply.
        # Do not fall back to local keyword heuristics — they caused accidental recordings when
        # the user sidetracked the conversation. If the model is not available or the
        # classifier fails to produce a definitive 'is_answer: true', we do NOT record.
        if not getattr(self, 'client', None):
            # No model available — do not record automatically. Let the conversation continue.
            return None
        try:
            questions = self.survey_manager.surveys.get(st.get('survey_id'), {}).get('questions', [])
            classifier = self._classify_survey_answer(questions, st, user_input)
            # Only accept an explicit model decision to record
            if not isinstance(classifier, dict) or not classifier.get('is_answer'):
                return None
            # Use normalized answer if model provided one
            if classifier.get('normalized') is not None:
                user_input = classifier.get('normalized')
        except Exception:
            # On any classifier error, be conservative: do not record
            return None

        # Record the response
        ok = self.survey_manager.record_response(session_id, waiting, user_input)
        # remove waiting flag (we'll either return a next question or completion text)
        st = self.survey_manager.get_state(session_id) or {}
        if st.get('status') == 'complete':
            try:
                st['completed_at'] = st.get('completed_at') or _now()
            except Exception:
                pass
            try:
                st['final_message_sent'] = True
            except Exception:
                pass
        try:
            st.pop("waiting_for", None)
        except Exception:
            pass
        if not ok:
            return None

        # If survey complete, return a completion message with score/interpretation
        try:
            if self.survey_manager.is_complete(session_id):
                score = self.survey_manager.compute_score(session_id)
                interpretation = score.get("interpretation")
                # Build a friendly completion summary that includes the recorded answers
                stext_lines = ["Thanks — I've recorded that. You completed the survey. Here are your responses:"]
                try:
                    # list each completed question with its recorded answer
                    survey_def = self.survey_manager.surveys.get(st.get('survey_id')) or {}
                    qmap = {q.get('question_id'): q for q in survey_def.get('questions', [])}
                    for qid, ans in st.get('responses', {}).items():
                        qtext = qmap.get(qid, {}).get('base_text') or qmap.get(qid, {}).get('text') or qid
                        stext_lines.append(f"- {qtext}: {ans}")
                except Exception:
                    pass
                if score.get("score") is not None:
                    stext_lines.append(f"\nYour {score.get('survey_id')} score is {score.get('score')}. {interpretation}")
                stext = "\n".join(stext_lines)
                # mark completed_at timestamp
                try:
                    st["completed_at"] = _now()
                except Exception:
                    pass
                # publish progress/completion event if available
                try:
                    from src.services.survey_events import publish_event
                    publish_event(session_id, {"type": "progress", "answered": len(st.get("responses", {})), "total": len(self.survey_manager.surveys.get(st.get("survey_id"), {}).get("questions", [])), "complete": True, "result": score})
                except Exception:
                    pass
                return stext
        except Exception:
            # continue to next-question logic on error
            pass

        # Otherwise, fetch the next question and return a combined ack + next question so the convo continues
        next_q = self.survey_manager.get_next_question(session_id, context=user_input)
        if next_q:
            # set the waiting_for marker so next user reply records against this question
            st["waiting_for"] = next_q.get("question_id")
            # publish interim progress
            try:
                from src.services.survey_events import publish_event
                publish_event(session_id, {"type": "progress", "answered": len(st.get("responses", {})), "total": len(self.survey_manager.surveys.get(st.get("survey_id"), {}).get("questions", [])), "complete": False})
            except Exception:
                pass
            # Friendly, conversational ack + natural follow-up question
            try:
                st["last_asked_at"] = time.time()
            except Exception:
                pass
            return f"Thanks — I appreciate you sharing that. That helps. One quick follow-up: {next_q.get('text')}"
        # Fallback acknowledgement
        try:
            from src.services.survey_events import publish_event
            publish_event(session_id, {"type": "progress", "answered": len(st.get("responses", {})), "total": len(self.survey_manager.surveys.get(st.get("survey_id"), {}).get("questions", [])), "complete": False})
        except Exception:
            pass
        return "Thanks — I've recorded that."

    # NOTE: heuristic-based answer detection has been removed. The LLM classifier
    # `_classify_survey_answer` is the single source of truth for deciding whether
    # a user reply answers a pending survey question. Do not reintroduce local
    # heuristics here.

    def _classify_survey_answer(self, questions, st, user_input: str) -> Optional[dict]:
        """Ask the LLM (if available) whether the user's reply answers the pending survey question.

        Returns a dict: {'is_answer': bool, 'normalized': Optional[str], 'reason': Optional[str]}
        Falls back to None on errors.
        """
        try:
            # Find pending question text
            qid = st.get('waiting_for')
            qtext = None
            if qid and questions:
                for q in questions:
                    if q.get('question_id') == qid:
                        qtext = q.get('text') or q.get('base_text')
                        break
            if not qtext:
                qtext = st.get('last_question_text') or ''

            # Build richer classifier prompt including question id and expected type/options
            qobj = None
            for q in (questions or []):
                if q.get('question_id') == qid:
                    qobj = q
                    break
            expected = qobj.get('expected_type') if qobj else None
            choices = qobj.get('response_options') if qobj else None

            # Prefer the exact phrasing last asked to the user for better paraphrase matching
            qtext = st.get('last_question_text') or qtext

            prompt_lines = [
                "You are a strict JSON-only classifier. DO NOT output any explanatory text or surrounding commentary.",
                "Task: Given a survey question (id + wording + expected response schema) and a user's reply, decide if the reply should be recorded as an answer to that question.",
                "Output schema (JSON only): {\"is_answer\": true|false, \"normalized\": null|\"...\", \"reason\": \"short explanation\"}",
                "Be conservative: if unsure, set is_answer to false. However, accept short natural paraphrases (e.g., 'I haven't', 'not really', 'nope') as valid answers for yes/no style questions.",
                "If the question expects a yes/no answer, normalize common negative/paraphrase replies to 'no' and positive to 'yes'.",
                "Respond ONLY with a single JSON object and nothing else. Use temperature 0.",
                "Examples (JSON only):",
                # Positive yes/no mapping
                json.dumps({"is_answer": True, "normalized": "no", "reason": "paraphrase"}, ensure_ascii=False),
                json.dumps({"is_answer": True, "normalized": "yes", "reason": "affirmative"}, ensure_ascii=False),
                # Non-answer / clarification
                json.dumps({"is_answer": False, "normalized": None, "reason": "clarifying_question"}, ensure_ascii=False),
                "",
                f"Question ID: {qid}",
                f"Question text: {qtext}",
            ]
            if expected:
                prompt_lines.append(f"Expected type: {expected}")
            if choices:
                prompt_lines.append(f"Options: {json.dumps(choices, ensure_ascii=False)}")
            prompt_lines.append("")
            prompt_lines.append(f"User reply: {user_input}")
            prompt = "\n".join(prompt_lines)
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a JSON classifier that decides if a user reply answers a survey question."},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = completion.choices[0].message.content or ''
            # Debug: if parsing fails or classifier returns None, log raw output for tuning
            try:
                debug_path = os.path.join(os.getcwd(), "uploads", "survey_classifier.log")
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                with open(debug_path, "a", encoding="utf-8") as dbg:
                    dbg.write(json.dumps({
                        "ts": int(time.time()),
                        "session": st.get('survey_id'),
                        "question_id": qid,
                        "question_text": qtext,
                        "user_reply": user_input,
                        "raw_output": txt[:2000]
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # Expect a single JSON object in the model output. Do not apply any
            # local heuristics — parse the JSON only. If parsing fails, return None
            # so the caller remains conservative.
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                return None
            js = m.group(0)
            data = json.loads(js)
            return {
                'is_answer': bool(data.get('is_answer')),
                'normalized': data.get('normalized'),
                'reason': data.get('reason') or None,
            }
        except Exception:
            return None

    def _maybe_inject_survey_question(self, session_id: str, context: str, msgs: List[Dict]) -> Optional[str]:
        """Return a natural variant for the next pending question or None."""
        if not getattr(self, "survey_manager", None):
            return None
        st = self.survey_manager.get_state(session_id)
        if not st or st.get("status") != "active":
            return None
        # If there's an outstanding question id (waiting_for), prefer that one so we can nudge
        waiting_qid = st.get("waiting_for")
        next_q = None
        if waiting_qid:
            # Try to find the matching question in the survey definition or pending list
            survey = self.survey_manager.surveys.get(st.get('survey_id')) or {}
            # check pending list first
            for q in st.get('pending', []):
                if q.get('question_id') == waiting_qid:
                    next_q = {"question_id": q["question_id"], "text": (q.get('natural_variants') or [q.get('base_text')])[0], "expected_type": q.get('expected_type')}
                    break
            # fallback to manager's next question
        if not next_q:
            next_q = self.survey_manager.get_next_question(session_id, context=context)
        if not next_q:
            return None
        # throttle so we don't repeat the same question
        last_assistant = None
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "")
                break
        if last_assistant and next_q.get("text") and next_q.get("text") in last_assistant:
            return None
        # mark that we've asked this question (but keep waiting_for so recording can match)
        try:
            st["waiting_for"] = next_q.get("question_id")
            # record the exact phrasing we asked so the classifier can compare user replies
            # against the same wording the user saw (helps with paraphrases).
            st["last_question_text"] = next_q.get("text")
            st["last_asked_at"] = time.time()
        except Exception:
            pass
        return next_q.get("text")

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
        # NOTE: Survey recording is now handled by the model itself. We do NOT pre-check
        # or record here via heuristics. Instead, if there's a pending survey question,
        # we will instruct the model (below) to append a short JSON marker indicating
        # whether the user's reply answers the pending question; that marker will be
        # parsed and recorded after the model's reply is received.

        # If awaiting a survey response, let the LLM-based classifier decide and
        # record it before generating the assistant reply. If a recording occurred,
        # _record_survey_response_if_expected will return an acknowledgement text
        # which we'll return immediately (mirrors the streaming path behavior).
        survey_ack = self._record_survey_response_if_expected(session_id, user_input)
        if survey_ack is not None:
            ack = _ensure_question(survey_ack)
            self.add_user(session_id, user_input)
            self.add_assistant(session_id, ack)
            return ack

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
        # If a survey is active and we haven't yet injected the survey context for this session,
        # provide the LLM with the full ordered list of survey questions as passive context so it
        # can naturally incorporate them into the conversation. This is a one-time injection per
        # survey session and avoids rigid or form-like instructions.
        try:
            st = None
            if getattr(self, 'survey_manager', None):
                st = self.survey_manager.get_state(session_id)
            if st and st.get('status') == 'active' and not st.get('context_injected'):
                survey_def = self.survey_manager.surveys.get(st.get('survey_id')) or {}
                qlist = survey_def.get('questions', [])
                if qlist:
                    # Clear behavior instructions so the model asks the survey questions
                    # in order, one at a time, waits for user reply, and includes a
                    # short progress marker like (Q 1/5). Do NOT evaluate or score answers;
                    # the server will record responses. After the final question, close the
                    # survey and do not ask further survey questions.
                    lines = [
                        f"Survey context: the user has selected '{survey_def.get('title', st.get('survey_id'))}'.",
                        "Behavior instructions:",
                        "- Ask the survey questions in the order provided, one question at a time.",
                        "- After each question include a short progress marker in parentheses, e.g. (Q 2/5).",
                        "- Wait for the user to reply before asking the next question. Do not assume answers.",
                        "- Do NOT attempt to score, record, or interpret answers; the server will handle recording.",
                        "- When all questions are asked and answered, thank the user, provide a brief acknowledgement, and stop asking survey questions.",
                        "Questions (do not ask them all at once; present one per turn):",
                    ]
                    for i, q in enumerate(qlist, start=1):
                        text = q.get('base_text') or (q.get('natural_variants') or [None])[0] or ''
                        lines.append(f"{i}) {text}")
                    ctx = "\n".join(lines)
                    msgs.insert(1, {"role": "system", "content": ctx})
                    try:
                        st['context_injected'] = True
                    except Exception:
                        pass
            # don't inject if we've already sent the final completion message
            if st and st.get('status') == 'complete':
                # ensure we tell the model not to ask further survey questions
                try:
                    stop_sys = (
                        "Survey Completed: The survey for this session is finished. Do NOT ask or prompt any further survey questions."
                    )
                    msgs.insert(1, {"role": "system", "content": stop_sys})
                except Exception:
                    pass
        except Exception:
            pass

        # Perform model call with resilience: try primary client, on failure attempt Groq fallback
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=self._temp(temperature),
                messages=msgs,
            )
        except Exception as e:
            err_str = str(e)
            print(f"[llm] primary provider {getattr(self,'provider',None)} call failed: {e}")
            # Detect common model-not-found responses and raise a helpful error
            if "model_not_found" in err_str or "does not exist" in err_str or "model `" in err_str and "not" in err_str:
                suggestion = os.getenv("GROQ_MODEL") or self.model
                msg = (
                    f"Groq reported model not found for model '{suggestion}'.\n"
                    "Action: set the environment variable GROQ_MODEL to a model you have access to (e.g. 'gpt-4o') or pick a supported model from your Groq dashboard.\n"
                    "If you want me to try a fallback model automatically, set GROQ_MODEL to the desired fallback.\n"
                    f"Original error: {err_str}"
                )
                raise RuntimeError(msg)
            # Try to fallback to Groq if available and configured (only if we haven't already selected Groq)
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and GroqClient is not None and self.provider != "Groq":
                try:
                    print("[llm] attempting fallback to GroqClient")
                    self.client = GroqClient(api_key=groq_key)
                    self.provider = "Groq"
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self._temp(temperature),
                        messages=msgs,
                    )
                except Exception as e2:
                    print(f"[llm] groq fallback also failed: {e2}")
                    if os.getenv("AVATAR_DEBUG") == "1":
                        traceback.print_exc()
                    raise
            else:
                # No groq key or already tried; re-raise to let caller handle
                raise
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
        # If awaiting a survey response, handle recording and short ack path (dev streaming simplified)
        survey_ack = self._record_survey_response_if_expected(session_id, user_input)
        if survey_ack is not None:
            ack = _ensure_question(survey_ack)
            self.add_user(session_id, user_input)
            self.add_assistant(session_id, ack)
            yield ack
            return

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
        # One-time passive survey context injection for streaming path as well
        try:
            st = None
            if getattr(self, 'survey_manager', None):
                st = self.survey_manager.get_state(session_id)
            if st and st.get('status') == 'active' and not st.get('context_injected'):
                survey_def = self.survey_manager.surveys.get(st.get('survey_id')) or {}
                qlist = survey_def.get('questions', [])
                if qlist:
                    lines = [f"Survey context: the user has selected '{survey_def.get('title', st.get('survey_id'))}'.",
                             "Please treat the following as contextual questions the user may answer or the assistant may naturally ask:"]
                    for i, q in enumerate(qlist, start=1):
                        text = q.get('base_text') or (q.get('natural_variants') or [None])[0] or ''
                        lines.append(f"{i}) {text}")
                    ctx = "\n".join(lines)
                    msgs.insert(1, {"role": "system", "content": ctx})
                    try:
                        st['context_injected'] = True
                    except Exception:
                        pass
        except Exception:
            pass
        # If the survey has become complete, insert a short system instruction
        # to prevent the model from asking any further survey questions.
        try:
            st2 = None
            if getattr(self, 'survey_manager', None):
                st2 = self.survey_manager.get_state(session_id)
            if st2 and st2.get('status') == 'complete' and st2.get('context_injected'):
                stop_sys = (
                    "Survey Completed: The survey for this session is finished. Do NOT ask or prompt any further survey questions."
                )
                msgs.insert(1, {"role": "system", "content": stop_sys})
        except Exception:
            pass
        if intent_context:
            msgs.insert(1, {"role": "system", "content": intent_context})

        acc = ""
        try:
            stream_ctx = self.client.chat.completions.stream(
                model=self.model,
                temperature=self._temp(temperature),
                messages=msgs,
            )
        except Exception as e:
            err_str = str(e)
            print(f"[llm] streaming call failed: {e}")
            if "model_not_found" in err_str or "does not exist" in err_str or ("model `" in err_str and "not" in err_str):
                suggestion = os.getenv("GROQ_MODEL") or self.model
                raise RuntimeError(
                    f"Groq reported model not found for model '{suggestion}'.\n"
                    "Action: set the environment variable GROQ_MODEL to a model you have access to (e.g. 'gpt-4o') or pick a supported model from your Groq dashboard.\n"
                    f"Original error: {err_str}"
                )
            raise

        with stream_ctx as stream:
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
                    err_text = f"[{self.provider} error: {getattr(event, 'error', 'unknown')}]"
                    self.add_assistant(session_id, err_text)
                    yield err_text
                    return

    # ---------- config helpers ----------
    def set_system_prompt(self, text: str) -> None:
        self.system_prompt = text

    def _temp(self, t: Optional[float]) -> float:
        return float(self.default_temperature if t is None else t)

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
                "- Consider a baseline checkup if symptoms persist.\n"
                "- 30 minutes of movement daily; consistent sleep.\n"
                "- Upload genomic data for rsID insights.\n"
                "- You’re making progress—small steps compound.\n\n"
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
    - Clinical
    - Lifestyle
    - Genomic (if any rsIDs mentioned; otherwise note pending)
    - Tone/Coaching
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
