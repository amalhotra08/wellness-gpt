from flask import Blueprint, Response, request, stream_with_context, jsonify
from src.services.llm import LlmBroker, stream_tokens
from src.utils.sessions import get_session, new_session
import os, json

bp = Blueprint('chat', __name__)
BROKER = LlmBroker(model="gpt-4o-mini")

def sse(event, data): return f"event: {event}\ndata: {data}\n\n"

@bp.post('/start')
def start_session():
    return jsonify({"session_id": new_session()})

@bp.post('/stream')
def stream_chat():
    payload = request.get_json(force=True)
    sid = payload.get('session_id'); text = (payload.get('message') or "").strip()
    _ = get_session(sid)  # keep your session store aligned

    def generate():
        # stream tokens
        msgs = BROKER._msgs(sid)
        msgs.append({"role":"user","content": text})
        for tok in stream_tokens(BROKER.client, BROKER.model, msgs):
            yield sse('token', tok)
        # finalize memory with full message from final event if you want (or just call reply())
        yield sse('final', '')

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
