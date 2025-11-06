import queue
import json

# Simple in-memory per-session queue map for SSE subscribers.
_queues = {}


def get_queue(session_id: str):
    q = _queues.get(session_id)
    if q is None:
        q = queue.Queue()
        _queues[session_id] = q
    return q


def publish_event(session_id: str, payload: dict):
    """Publish a JSON-serializable payload to the session queue.
    If no queue exists yet, create it. Non-blocking.
    """
    try:
        q = get_queue(session_id)
        # place a JSON-serializable copy
        q.put_nowait(payload)
    except Exception:
        pass
