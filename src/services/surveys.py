import json
import os
import random
from typing import Dict, List, Optional, Any
import time
import json
import os


class SurveyManager:
    """Load surveys from JSON and maintain per-session active survey state.

    Minimal implementation to support conversational injection and recording.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.path.dirname(__file__), "surveys.json")
        self.surveys = self._load_surveys()
        # session_id -> active survey state
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _load_surveys(self) -> Dict[str, Dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {s["id"]: s for s in data.get("surveys", [])}
        except Exception:
            return {}

    def list_surveys(self) -> List[Dict]:
        return [
            {"id": s["id"], "title": s.get("title"), "description": s.get("description"), "goal_tag": s.get("goal_tag")}
            for s in self.surveys.values()
        ]

    def start_survey(self, session_id: str, survey_id: str) -> bool:
        s = self.surveys.get(survey_id)
        if not s:
            return False
        # Build pending questions list (shallow copy)
        pending = [q.copy() for q in s.get("questions", [])]
        self.sessions[session_id] = {
            "survey_id": survey_id,
            "started_at": int(time.time()),
            "completed_at": None,
            "responses": {},
            "pending": pending,
            "completed": [],
            "status": "active",
            # mark that we haven't injected the passive survey context into the LLM yet
            "context_injected": False,
        }
        return True

    def get_state(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def _choose_variant(self, q: Dict, context: str = "") -> str:
        variants = q.get("natural_variants") or []
        if not variants:
            return q.get("base_text")
        # Basic heuristic: prefer a variant containing a keyword from context
        lc = (context or "").lower()
        for v in variants:
            for w in re_keywords(v):
                if w in lc:
                    return v
        return random.choice(variants)

    def get_next_question(self, session_id: str, context: str = "") -> Optional[Dict]:
        st = self.sessions.get(session_id)
        if not st or st.get("status") != "active":
            return None
        if not st.get("pending"):
            return None
        # Return the next pending question as a dict with chosen variant
        q = st["pending"][0]
        variant = None
        if q.get("natural_variants"):
            variant = random.choice(q.get("natural_variants"))
        else:
            variant = q.get("base_text")
        return {"question_id": q["question_id"], "text": variant, "expected_type": q.get("expected_type")}

    def record_response(self, session_id: str, question_id: str, answer: Any) -> bool:
        st = self.sessions.get(session_id)
        if not st:
            return False
        # find pending question
        for i, q in enumerate(st.get("pending", [])):
            if q.get("question_id") == question_id:
                st["responses"][question_id] = answer
                st["completed"].append(q)
                st["pending"].pop(i)
                # check completion criteria
                survey = self.surveys.get(st["survey_id"]) if st.get("survey_id") else None
                criteria = survey.get("completion_criteria") if survey else None
                if criteria:
                    # simple: if pending empty or response count >= criteria
                    try:
                        needed = int(criteria)
                    except Exception:
                        needed = None
                    if needed is not None and len(st["responses"]) >= needed:
                            st["status"] = "complete"
                            try:
                                st["completed_at"] = int(time.time())
                            except Exception:
                                st["completed_at"] = None
                    if st.get("status") == "complete":
                        # Persist completed response
                        try:
                            self._persist_session(session_id)
                        except Exception:
                            pass
                    # Additional safety: if we've recorded as many responses as questions,
                    # mark complete to avoid infinite loops.
                    try:
                        total_q = len(self.surveys.get(st.get('survey_id'), {}).get('questions', []))
                        if total_q and len(st.get('responses', {})) >= total_q:
                            st['status'] = 'complete'
                            st['completed_at'] = st.get('completed_at') or int(time.time())
                            try:
                                self._persist_session(session_id)
                            except Exception:
                                pass
                    except Exception:
                        pass
                if not st.get("pending"):
                    st["status"] = "complete"
                    try:
                        st["completed_at"] = int(time.time())
                    except Exception:
                        pass
                    try:
                        self._persist_session(session_id)
                    except Exception:
                        pass
                return True
        return False

    def is_complete(self, session_id: str) -> bool:
        st = self.sessions.get(session_id)
        if not st:
            return False
        return st.get("status") == "complete"

    def compute_score(self, session_id: str) -> Dict[str, Any]:
        # Very small scoring helpers for PSQ/PHQ-9/GAD-7 if present (best-effort)
        st = self.sessions.get(session_id) or {}
        survey_id = st.get("survey_id")
        resp = st.get("responses", {})
        out = {"survey_id": survey_id, "score": None, "interpretation": None}
        if not survey_id:
            return out
        if survey_id.startswith("phq"):
            # sum numeric responses (expect 0-3)
            try:
                s = sum(int(v) for v in resp.values())
                out["score"] = s
                out["interpretation"] = phq9_interpretation(s)
            except Exception:
                pass
        elif survey_id.startswith("gad"):
            try:
                s = sum(int(v) for v in resp.values())
                out["score"] = s
                out["interpretation"] = gad7_interpretation(s)
            except Exception:
                pass
        elif survey_id.startswith("psq"):
            # PSQ-like scoring (example): average frequency
            out["score"] = None
            # PSQ does not produce a numeric interpretation here; leave empty to avoid
            # repeating 'Survey complete' in the UI. The caller/UI will render a suitable
            # completion message instead.
            out["interpretation"] = ""
        return out

    def _persist_session(self, session_id: str) -> None:
        """Append session survey result to uploads/survey_results.jsonl for later review."""
        st = self.sessions.get(session_id) or {}
        if not st:
            return
        survey_id = st.get("survey_id")
        resp = st.get("responses", {})
        result = self.compute_score(session_id)
        record = {
            "ts": int(time.time()),
            "session_id": session_id,
            "survey_id": survey_id,
            "responses": resp,
            "result": result,
        }
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", "survey_results.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def re_keywords(text: str) -> List[str]:
    # simple tokenization for heuristic matching
    return [w.lower() for w in re_split_words(text) if len(w) > 3]


def re_split_words(text: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z]{3,}", text or "")


def phq9_interpretation(score: int) -> str:
    if score <= 4:
        return "Minimal or no depression"
    if score <= 9:
        return "Mild depression"
    if score <= 14:
        return "Moderate depression"
    if score <= 19:
        return "Moderately severe depression"
    return "Severe depression"


def gad7_interpretation(score: int) -> str:
    if score <= 4:
        return "Minimal anxiety"
    if score <= 9:
        return "Mild anxiety"
    if score <= 14:
        return "Moderate anxiety"
    return "Severe anxiety"
