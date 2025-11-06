## Repo: WellnessGPT — quick orientation for AI coding agents

This file gives concise, actionable guidance so an AI coding assistant can be productive immediately in this codebase.

- Entry point: `app.py` — a Flask app that serves the single-page UI (`templates/index.html`) and JSON endpoints under `/api/*`.
  - Run locally: `python app.py` (the file contains `if __name__ == '__main__': app.run(..., use_reloader=False)`).
  - Note: `use_reloader=False` is intentional — in-memory session histories (LlmBroker) are lost if the process forks.

- Core service layer: `src/services/` (key modules)
  - `llm.py` — LlmBroker: in-memory session history, Groq adapter (uses `GROQ_API_KEY`), dev fallback when no key, always-attaches citations, survey integration (loads `surveys.json`).
  - `avatar.py` — avatar TTS/video rendering helpers (called by `/api/avatar` and `/api/tts`).
  - `expert_finder.py` — lightweight provider search helpers used by expert-finder UI card.
  - `surveys.py` & `surveys.json` — survey defs and SurveyManager (survey flow and server-side recording).

- Runtime configuration
  - Environment loaded via `python-dotenv` in `app.py` (`.env` is used). Important env vars: `GROQ_API_KEY`, `GROQ_MODEL`, `AVATAR_DEBUG`, `CITATIONS_V2_DISABLE`.
  - If `GROQ_API_KEY` is unset or `GroqClient` import fails, LlmBroker runs in dev mode and returns simple echo-like replies. Tests or feature dev can rely on that.

- Developer workflows & useful commands
  - Install deps: `pip install -r requirements.txt` (root). The `OpenFactVerification/` subproject has its own `pyproject.toml` and README — treat it as a separate component.
  - Start server: `python app.py` (dev) → UI at http://localhost:5000. Use the browser dev tools to inspect SSE `/api/chat/stream` and `/api/survey/stream` events.
  - Uploads: files saved to `uploads/` and served at `/uploads/<path>` via `get_upload`. Clean up uploaded artifacts during tests.

- Patterns & conventions to follow when modifying code
  - LlmBroker is stateful and stores per-session histories in-memory (keyed by cookie `sid`). Avoid code changes that create multiple processes or reloader-enabled runs unless you migrate state to DB.
  - Survey handling is LLM-driven: do not add local heuristics for deciding whether a reply answers a survey question — `llm.py` uses a model-based JSON-only classifier (`_classify_survey_answer`). If you change survey flow, update `src/services/surveys.py` and `surveys.json` consistently.
  - Citation logic: prefer `citations_v2` pipeline when available; fallback to legacy `citations.py`. Changes to citation behavior should update both `src/services/citations*` and where `LlmBroker._attach_citations` is invoked.

- Integration points & external dependencies
  - Groq (via `src/services/groq_client.py`) when `GROQ_API_KEY` present. Avoid hardcoding model names; prefer `GROQ_MODEL` env var.
  - edge-tts is used for TTS fallback; `moviepy`/`pydub` used in avatar/video pipeline (see `requirements.txt`).
  - `OpenFactVerification/` is a bundled subproject (separate README and poetry/pip instructions). Don't conflate its virtualenv with the root app — treat as optional plugin.

- Fast examples for code changes
  - Add a new API route that uses LlmBroker: import the global `BROKER` from `app.py`, then call `BROKER.reply_sync(sid, text)` or `BROKER.stream_reply(...)`. Respect cookie `sid` management via `_ensure_session_cookie`.
  - To add a long-running background job, avoid Flask reloader and prefer an external worker or explicit background thread started under `if __name__ == '__main__'` with care about process forking.

- Files to inspect when debugging unexpected behavior
  - `templates/index.html` (frontend logic + JS fetches `/api/*` endpoints)
  - `src/services/llm.py` (business logic: condensation, citations, survey integration)
  - `src/services/avatar.py` (avatar render pipeline)
  - `src/services/expert_finder.py` and front-end expert card (`templates/index.html`) for provider search flow
  - `requirements.txt` and `OpenFactVerification/README.md` for dependency differences

If anything above is unclear or you'd like more examples (unit tests, CI commands, or a short checklist for PRs touching LLM logic), tell me which section to expand and I'll iterate.  
