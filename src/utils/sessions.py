import uuid
from typing import Dict

# In-memory for dev; swap to Redis/db in prod
_sessions: Dict[str, dict] = {}

def new_session() -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {"memory": [], "created": True}
    return sid

def get_session(sid: str) -> dict:
    return _sessions.setdefault(sid, {"memory": []})