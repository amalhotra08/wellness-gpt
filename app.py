# app.py
import os
import time
import uuid
import secrets
from datetime import datetime, timedelta
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    make_response,
    redirect,
    flash,
)
from src.services.llm import LlmBroker  # uses your refined SYSTEM_PROMPT etc.
# from src.services.email_utils import send_email  # Email no longer used for summary download feature
from src.services.avatar import synth_and_render
from src.services.expert_finder import detect_expert_intent, search_providers
import asyncio
from flask import stream_with_context
from src.services import survey_events
import json
from dotenv import load_dotenv
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from sqlalchemy import inspect, text
from src.models import db, User, Conversation, Message, Consent

load_dotenv()   # loads .env into os.environ

import logging

# ---- Flask setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Vercel's /var/task filesystem is read-only.
# Force Flask's instance folder into /tmp, which is writable in serverless functions.
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
    instance_path="/tmp"
)

# Hardcoded for now. Not best practice, but avoids the production env var crash.
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-change-in-prod"
)

# Vercel cannot write to sqlite:///hms_gami.sqlite because that resolves under the app/instance path.
# Use /tmp so SQLAlchemy can create/write the SQLite file.
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/hms_gami.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Security: Session cookie hardening
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Optional but useful if upload code references app.config["UPLOAD_FOLDER"]
app.config["UPLOAD_FOLDER"] = "/tmp/uploads"

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "landing"

# Vercel can only write to /tmp
os.makedirs("/tmp/", exist_ok=True)

# Hardcoded defaults for now
SESSION_DURATION_SECONDS = 1800
PARTICIPANT_PIN_LENGTH = 6

def _generate_unique_participant_pin() -> str:
    """Return a random numeric participant PIN that is unique in the users table."""
    digits = max(4, PARTICIPANT_PIN_LENGTH)
    upper_bound = 10 ** digits
    for _ in range(50):
        candidate = f"{secrets.randbelow(upper_bound):0{digits}d}"
        if not User.query.filter_by(participant_pin=candidate).first():
            return candidate
    # Extremely unlikely fallback: include a short UUID-derived suffix if the PIN space is saturated.
    return f"{secrets.randbelow(upper_bound):0{digits}d}-{uuid.uuid4().hex[:4]}"

def _ensure_user_pin(user: User) -> str:
    """Backfill a participant PIN for older accounts and return it."""
    if not getattr(user, "participant_pin", None):
        user.participant_pin = _generate_unique_participant_pin()
        db.session.add(user)
        db.session.commit()
    return user.participant_pin

def _ensure_runtime_schema() -> None:
    """Small safety migration for local SQLite/db.create_all deployments."""
    inspector = inspect(db.engine)
    user_columns = {c["name"] for c in inspector.get_columns("users")}
    with db.engine.begin() as conn:
        if "participant_pin" not in user_columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN participant_pin VARCHAR(12)"))
        # SQLite allows multiple NULLs in a unique index, which is useful while old users are backfilled.
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_participant_pin ON users (participant_pin)"))

# Create tables within app context
with app.app_context():
    db.create_all()
    _ensure_runtime_schema()

# Single broker instance (swap for per-user if you add auth/DB later)
BROKER = LlmBroker()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def check_consent():
    """Block access to app if consent not given."""
    if request.endpoint in ('static', 'login', 'login_page', 'register', 'register_page', 'landing', 'logout', 'consent', 'api_consent', 'consent_page'):
        return
    if current_user.is_authenticated:
        # Check if user has consented
        if not current_user.consents:
            return redirect("/consent")

# ---- Helpers ----
def _now() -> str:
    return time.ctime(time.time())

def _initial_greeting() -> str:
    """First assistant message shown/saved for the participant."""
    pin_line = ""
    if current_user.is_authenticated:
        pin = _ensure_user_pin(current_user)
        pin_line = (
            f"\n\nYour participant PIN code is: {pin}. "
            "Please keep this code so questionnaire responses can be matched with your session engagement metrics."
        )
    return "Hello! I'm your AI health assistant. How can I help you with your wellness journey today?" + pin_line

def _conversation_for_user():
    if not current_user.is_authenticated:
        return None
    return Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).first()

def _session_timer_payload(session_id=None) -> dict:
    """Return server-authoritative countdown metadata for the active conversation."""
    conv = Conversation.query.get(session_id) if session_id else _conversation_for_user()
    if not conv:
        return {
            "duration_seconds": SESSION_DURATION_SECONDS,
            "remaining_seconds": SESSION_DURATION_SECONDS,
            "expired": False,
        }
    start_at = conv.created_at or datetime.utcnow()
    expires_at = start_at + timedelta(seconds=SESSION_DURATION_SECONDS)
    remaining = max(0, int((expires_at - datetime.utcnow()).total_seconds()))
    return {
        "duration_seconds": SESSION_DURATION_SECONDS,
        "remaining_seconds": remaining,
        "expired": remaining <= 0,
        "started_at": start_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

def _session_expired(session_id=None) -> bool:
    return bool(_session_timer_payload(session_id).get("expired"))

def _data_for_render(session_id: str):
    """
    Template data with existing history or a default greeting.
    """
    hist = BROKER.get_history(session_id)
    if not hist:
        return {
            "messages": [
                {"role": "assistant", "content": _initial_greeting(), "time": _now()}
            ],
            "summary": "",
        }
    return {"messages": hist, "summary": ""}

def _ensure_session_cookie(resp=None):
    """
    Ensure the browser has a 'sid' cookie corresponding to the user's conversation.
    Only called for authenticated users (index route).
    """
    if not current_user.is_authenticated:
        # Should be covered by @login_required but safe guard
        return None, resp

    sid = request.cookies.get("sid")
    # Find active conversation for user
    conv = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.updated_at.desc()).first()
    
    if not conv:
        new_id = uuid.uuid4().hex
        conv = Conversation(id=new_id, user_id=current_user.id, title="New Chat")
        db.session.add(conv)
        db.session.commit()
    
    if sid != conv.id:
        # Update cookie to match conversation ID
        if resp is None:
            resp = make_response()
        resp.set_cookie("sid", conv.id, max_age=60 * 60 * 24 * 365, httponly=False, samesite="Lax")
        return conv.id, resp
    
    return sid, resp

# ---- Routes ----
# ---- Routes ----

@app.route("/landing")
def landing():
    if current_user.is_authenticated:
        return redirect("/") # Go to app if logged in
    return render_template("landing.html")

@app.route("/login", methods=["GET"])
def login_page():
    if current_user.is_authenticated:
        return redirect("/")
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    if current_user.is_authenticated:
        return redirect("/")
    return render_template("register.html")

@app.route("/api/auth/register", methods=["POST"])
def register():
    username = request.form.get("username")
    password = request.form.get("password")
    
    if not username or not password:
        flash("Username and password required")
        return redirect("/register")
    
    if User.query.filter_by(username=username).first():
        flash("Username already exists")
        return redirect("/register")
        
    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        participant_pin=_generate_unique_participant_pin(),
    )
    db.session.add(user)
    db.session.commit()
    
    login_user(user)
    return redirect("/consent")

@app.route("/consent")
@login_required
def consent_page():
    if current_user.consents:
        return redirect("/")
    return render_template("consent.html")

@app.post("/api/consent")
@login_required
def api_consent():
    # Record consent
    c = Consent(user_id=current_user.id, version="1.0", ip_hash=None) # TODO: hash IP if needed
    db.session.add(c)
    db.session.commit()
    return redirect("/")

@app.route("/api/auth/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        _ensure_user_pin(user)
        login_user(user)
        return redirect("/")
        
    flash("Invalid credentials")
    return redirect("/login")

@app.route("/api/auth/logout")
@login_required
def logout():
    logout_user()
    resp = make_response(redirect("/landing"))
    resp.delete_cookie("sid")
    return resp

def _index_template_context(session_id: str, play_initial_greeting: bool = False) -> dict:
    timer = _session_timer_payload(session_id)
    return {
        "data": _data_for_render(session_id),
        "initial_greeting": _initial_greeting(),
        "play_initial_greeting": play_initial_greeting,
        "session_timer": timer,
        "session_duration_seconds": timer.get("duration_seconds", SESSION_DURATION_SECONDS),
        "session_remaining_seconds": timer.get("remaining_seconds", SESSION_DURATION_SECONDS),
        "session_expired": timer.get("expired", False),
        "participant_pin": _ensure_user_pin(current_user),
    }

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # Keep the same session between requests via cookie.
    if request.method == "POST":
        sid, _ = _ensure_session_cookie()
        text_value = (request.form.get("user_input") or "").strip()
        if text_value and not _session_expired(sid):
            BROKER.reply_sync(sid, text_value)
        resp = make_response(render_template("index.html", **_index_template_context(sid)))
        _, resp = _ensure_session_cookie(resp)
        return resp

    # GET: set a conversation cookie, seed the first GPT greeting once, then render history.
    resp = make_response()
    sid, resp = _ensure_session_cookie(resp)
    play_initial = False
    if sid and BROKER.history_size(sid) == 0:
        BROKER.add_assistant(sid, _initial_greeting())
        play_initial = True
    resp.set_data(render_template("index.html", **_index_template_context(sid, play_initial_greeting=play_initial)))
    return resp


@app.get("/api/session/timer")
@login_required
def api_session_timer():
    """Return the authoritative 30-minute session countdown for the current conversation."""
    resp = make_response()
    sid, resp = _ensure_session_cookie(resp)
    payload = {"ok": True, **_session_timer_payload(sid)}
    resp.set_data(json.dumps(payload))
    resp.mimetype = "application/json"
    return resp

# ---- JSON chat (sync) ----
@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    sid = request.cookies.get("sid") or "default"
    user_input = (data.get("user_input") or "").strip()
    if not user_input:
        return jsonify({"reply": "Please send a message."}), 400

    if current_user.is_authenticated and _session_expired(sid):
        return jsonify({
            "reply": "The 30-minute session has ended. Thank you for participating.",
            "history_size": len(BROKER.get_history(sid)),
            "session_expired": True,
            "remaining_seconds": 0,
        })

    low = user_input.lower()
    wants_cites = any(k in low for k in ("source", "citation", "prove", "link", "references", "evidence"))
    # Detect expert intent before generating reply (so future adaptation can occur if desired)
    intent = detect_expert_intent(user_input)
    intent_ctx = None
    if intent:
        print(f"[expert_intent] session={sid} label={intent['label']} taxonomy={intent['taxonomy']}")
        # Provide transient context so model acknowledges capability without re-performing search itself.
        intent_ctx = (
            f"Expert Finder Context: The user appears to be requesting help locating a {intent['label']} (taxonomy: {intent['taxonomy']}). "
            "You have an auxiliary tool that will surface a separate 'Expert Finder' card in the UI handling location & provider lookup. "
            "Do NOT claim you cannot search. Instead: briefly acknowledge the specialty, offer 1-2 screening questions or prep steps (e.g., symptoms to note, records to gather), and invite them to use the card that appears. Keep it short before your usual guidance."
        )
    reply = BROKER.reply_sync(sid, user_input, force_citations=wants_cites, intent_context=intent_ctx)
    timer_payload = _session_timer_payload(sid) if current_user.is_authenticated else {}
    return jsonify({
        "reply": reply,
        "history_size": len(BROKER.get_history(sid)),
        "expert_intent": intent,
        "session_expired": timer_payload.get("expired", False),
        "remaining_seconds": timer_payload.get("remaining_seconds"),
    })


@app.get("/api/surveys")
def api_list_surveys():
    """Return available surveys loaded by the broker's SurveyManager."""
    sid = request.cookies.get("sid") or "default"
    mgr = getattr(BROKER, "survey_manager", None)
    if not mgr:
        return jsonify({"surveys": []})
    return jsonify({"surveys": mgr.list_surveys()})


@app.post("/api/survey/select")
def api_select_survey():
    j = request.get_json(silent=True) or {}
    survey_id = (j.get("survey_id") or "").strip()
    sid = request.cookies.get("sid") or "default"
    mgr = getattr(BROKER, "survey_manager", None)
    if not mgr:
        return jsonify({"ok": False, "error": "surveys unavailable"}), 500
    if not survey_id:
        return jsonify({"ok": False, "error": "survey_id missing"}), 400
    if current_user.is_authenticated and _session_expired(sid):
        return jsonify({"ok": False, "error": "session timer expired", "session_expired": True}), 403
    ok = mgr.start_survey(sid, survey_id)
    resp_data = {"ok": ok, "survey_id": survey_id}
    # If started successfully, generate an assistant starter message + first question and return it
    # Send a brief, natural starter message so the UI shows the survey has begun.
    # We intentionally do NOT send the first survey question verbatim here; the
    # LLM is given the full survey questions as passive context and will ask the
    # questions naturally during the conversation. This keeps the tone conversational
    # while giving an immediate cue to the user that the survey started.
    if ok:
        try:
            survey = mgr.surveys.get(survey_id, {})
            title = survey.get('title') or 'short survey'
            # Get the first question so the UI and user have an explicit starting point.
            first = mgr.get_next_question(sid, context="")
            if first:
                # set waiting marker and add assistant message (starter + question)
                st = mgr.get_state(sid)
                st["waiting_for"] = first.get("question_id")
                st["last_question_text"] = first.get("text")
                starter = f"Okay — I've started the {title}. I'll ask a few quick, conversational questions. There are no right or wrong answers. {first.get('text') }"
                BROKER.add_assistant(sid, starter)
                resp_data.update({
                    "assistant_message": starter,
                    "first_question": first.get("text"),
                    "waiting_for": first.get("question_id"),
                })
            else:
                starter = f"Okay — I've started the {title}. I'll ask a few quick, conversational questions. There are no right or wrong answers."
                BROKER.add_assistant(sid, starter)
                resp_data.update({"assistant_message": starter})
        except Exception:
            pass
    return jsonify(resp_data)


@app.get("/api/survey/status")
def api_survey_status():
    sid = request.cookies.get("sid") or "default"
    mgr = getattr(BROKER, "survey_manager", None)
    if not mgr:
        return jsonify({"ok": False, "error": "surveys unavailable"}), 500
    st = mgr.get_state(sid) or {}
    # derive progress
    survey_id = st.get("survey_id")
    survey = mgr.surveys.get(survey_id) if survey_id else None
    total_q = 0
    if survey:
        total_q = len(survey.get("questions", []))
    answered = len(st.get("responses", {}))
    pending = len(st.get("pending", []))
    complete = st.get("status") == "complete"
    resp = {
        "ok": True,
        "survey_id": survey_id,
        "complete": complete,
        "answered": answered,
        "pending": pending,
        "total": total_q,
        "state": st,
    }
    if complete:
        try:
            resp["result"] = mgr.compute_score(sid)
        except Exception:
            resp["result"] = None
    return jsonify(resp)


@app.get("/api/survey/responses")
def api_survey_responses():
    sid = request.cookies.get("sid") or "default"
    mgr = getattr(BROKER, "survey_manager", None)
    if not mgr:
        return jsonify({"ok": False, "error": "surveys unavailable"}), 500
    st = mgr.get_state(sid) or {}
    return jsonify({"ok": True, "responses": st.get("responses", {}), "state": st})


@app.get("/api/survey/download")
def api_survey_download():
    sid = request.cookies.get("sid") or "default"
    mgr = getattr(BROKER, "survey_manager", None)
    if not mgr:
        return jsonify({"ok": False, "error": "surveys unavailable"}), 500
    st = mgr.get_state(sid) or {}
    record = {
        "participant_pin": _ensure_user_pin(current_user) if current_user.is_authenticated else None,
        "session_id": sid,
        "survey_id": st.get("survey_id"),
        "responses": st.get("responses", {}),
        "state": st,
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    resp = make_response(payload)
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Content-Disposition"] = f"attachment; filename=survey_{sid}.json"
    return resp


@app.get('/api/survey/stream')
def api_survey_stream():
    """Server-Sent Events stream for survey progress for the current session (cookie 'sid')."""
    sid = request.cookies.get('sid') or 'default'
    mgr = getattr(BROKER, 'survey_manager', None)
    if not mgr:
        return jsonify({'error': 'surveys unavailable'}), 500

    q = survey_events.get_queue(sid)

    def gen():
        # send initial snapshot
        try:
            st = mgr.get_state(sid) or {}
            total = 0
            survey_id = st.get('survey_id')
            if survey_id:
                survey = mgr.surveys.get(survey_id) or {}
                total = len(survey.get('questions', []))
            init = {'type': 'progress', 'answered': len(st.get('responses', {})), 'total': total, 'complete': st.get('status') == 'complete', 'state': st}
            yield f"event: progress\ndata: {json.dumps(init)}\n\n"
        except Exception:
            pass

        while True:
            try:
                item = q.get(timeout=30)
                try:
                    yield f"event: progress\ndata: {json.dumps(item)}\n\n"
                except Exception:
                    continue
            except Exception:
                # keepalive comment
                yield ": keepalive\n\n"

    return Response(stream_with_context(gen()), mimetype='text/event-stream')


# Debug helper (temporary) -- returns broker introspection
@app.get('/_debug_broker')
def _debug_broker():
    try:
        import inspect
        info = {
            'type': str(type(BROKER)),
            'has_history_size': hasattr(BROKER, 'history_size'),
            'methods': [n for n in dir(BROKER) if not n.startswith('_')],
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    resp = {"state": st}
    if mgr.is_complete(sid):
        resp["result"] = mgr.compute_score(sid)
    return jsonify(resp)


# ---- JSON chat (SSE streaming) ----
@app.post("/api/chat/stream")
def stream_chat():
    data = request.get_json(silent=True) or {}
    sid = request.cookies.get("sid") or "default"
    message = (data.get("message") or "").strip()
    if not message:
        return Response("event: final\ndata: missing message\n\n", mimetype="text/event-stream")

    if current_user.is_authenticated and _session_expired(sid):
        return Response(
            "event: token\ndata: The 30-minute session has ended. Thank you for participating.\n\n"
            "event: final\ndata: session_expired\n\n",
            mimetype="text/event-stream",
        )

    low = message.lower()
    wants_cites = any(k in low for k in ("source", "citation", "prove", "link", "references", "evidence"))

    # Mirror intent detection for streaming path
    intent = detect_expert_intent(message)
    intent_ctx = None
    if intent:
        print(f"[expert_intent] session={sid} label={intent['label']} taxonomy={intent['taxonomy']}")
        intent_ctx = (
            f"Expert Finder Context: The user appears to be requesting help locating a {intent['label']} (taxonomy: {intent['taxonomy']}). "
            "A separate UI card will handle the provider search. Acknowledge capability; provide brief prep advice; don't say you cannot search."
        )

    def gen():
        try:
            for tok in BROKER.stream_reply(sid, message, force_citations=wants_cites, intent_context=intent_ctx):
                yield f"event: token\ndata: {tok}\n\n"
            yield "event: final\ndata: done\n\n"
        except Exception as e:
            yield f"event: final\ndata: ERROR {e}\n\n"

    return Response(gen(), mimetype="text/event-stream")


# ---- Helper endpoints you referenced ----
@app.post("/username-endpoint")
def username_endpoint():
    j = request.get_json(silent=True) or {}
    username = (j.get("username") or "").strip()
    if not username:
        return jsonify({"message": "username missing"}), 400
    os.makedirs(os.path.join("/tmp/uploads", username), exist_ok=True)
    return jsonify({"message": "ok"})

@app.post("/generate_image")
def generate_image():
    # Stub: wire to image generation later
    prompt = request.form.get("user_input", "")
    return jsonify({"ok": True, "prompt": prompt})

@app.post("/api/upload/genomic")
def upload_genomic():
    f = request.files.get("genomicFile")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    save_path = os.path.join("/tmp/uploads", f.filename)
    os.makedirs("/tmp/uploads", exist_ok=True)
    f.save(save_path)
    # TODO: call compare_gene_conditions(save_path, ...) and return real results
    return jsonify({"uploaded": True, "records": [], "count": 0})

# --- New: Downloadable Summary Endpoint (HTML file) ---
@app.get("/api/summary/download")
def summary_download():
    """Return a styled HTML file (as attachment) containing the session summary."""
    sid = request.cookies.get("sid") or request.args.get("session_id", "default").strip()
    summary_md = BROKER.session_summary(sid)

    # Minimal markdown -> HTML transform (headings, bullets, bold). Keeps things simple; avoids extra deps.
    def md_to_html(md: str) -> str:
        lines = md.splitlines()
        html_lines = []
        in_list = False
        for line in lines:
            raw = line.rstrip()
            if raw.startswith("### "):
                if in_list: html_lines.append("</ul>"); in_list = False
                html_lines.append(f"<h3>{raw[4:]}</h3>")
                continue
            if raw.startswith("## "):
                if in_list: html_lines.append("</ul>"); in_list = False
                html_lines.append(f"<h2>{raw[3:]}</h2>")
                continue
            if raw.startswith("# "):
                if in_list: html_lines.append("</ul>"); in_list = False
                html_lines.append(f"<h1>{raw[2:]}</h1>")
                continue
            if raw.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>"); in_list = True
                html_lines.append(f"<li>{raw[2:]}</li>")
                continue
            # Blank line
            if raw.strip() == "":
                if in_list: html_lines.append("</ul>"); in_list = False
                html_lines.append("<br>")
                continue
            # Bold **text**
            safe = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\\1</strong>", safe)
            if in_list:
                # treat as paragraph inside list? close list first
                html_lines.append("</ul>"); in_list = False
            html_lines.append(f"<p>{safe}</p>")
        if in_list: html_lines.append("</ul>")
        return "\n".join(html_lines)

    import re  # local import (regex used in conversion)
    body_html = md_to_html(summary_md)
    ts = int(time.time())
    title = f"Chat System A Session Summary"
    html = f"""<!DOCTYPE html><html lang='en'>
<head>
<meta charset='utf-8'>
<title>{title}</title>
<style>
  body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; margin:40px; color:#1a1a1a; line-height:1.55; }}
  h1,h2,h3 {{ font-weight:600; line-height:1.2; }}
  h1 {{ font-size:28px; margin-top:0; }}
  h2 {{ font-size:20px; margin-top:32px; border-bottom:1px solid #eee; padding-bottom:4px; }}
  h3 {{ font-size:16px; margin-top:24px; }}
  ul {{ padding-left:22px; }}
    li {{ margin:6px 0; }}
    p {{ margin:12px 0; }}
  code, pre {{ background:#f5f5f5; padding:2px 4px; border-radius:4px; }}
  .meta {{ font-size:12px; color:#666; margin-top:40px; }}
  .badge {{ display:inline-block; background:#2563eb; color:#fff; padding:4px 10px; border-radius:20px; font-size:11px; letter-spacing:.5px; text-transform:uppercase; }}
  .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:30px; }}
  .footer-note {{ font-size:11px; color:#999; margin-top:60px; text-align:center; }}
</style>
</head>
<body>
 <div class='header'>
   <div>
     <h1 style='margin:0 0 4px 0;'>Session Summary</h1>
     <div class='badge'>Chat System A</div>
   </div>
   <div style='text-align:right;font-size:12px;color:#555;'>Participant PIN: {_ensure_user_pin(current_user) if current_user.is_authenticated else 'N/A'}<br>Session: {sid}<br>Generated: {time.ctime(time.time())}</div>
 </div>
 {body_html}
 <div class='footer-note'>Generated by Chat System A – AI wellness companion.</div>
</body></html>"""

    # Offer as downloadable file
    from flask import make_response
    resp = make_response(html)
    filename = f"summary_{sid}_{ts}.html"
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return resp

@app.route('/api/summary', methods=['GET'])
def get_summary():
    # Get the session id from query params or default
    sid = request.args.get('session_id', 'default').strip()
    # Get the summary from the broker/session
    summary = BROKER.session_summary(sid)
    return jsonify({"summary": summary})

# ---- Expert Finder (Phase 1: research only) ----
@app.post("/api/expert/intent")
def expert_intent():
    j = request.get_json(silent=True) or {}
    text = (j.get("text") or "").strip()
    meta = detect_expert_intent(text)
    return jsonify({"intent": meta})

@app.post("/api/expert/search")
def expert_search():
    j = request.get_json(silent=True) or {}
    try:
        lat = float(j.get("lat"))
        lng = float(j.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid lat/lng"}), 400
    tax = (j.get("taxonomy") or "").strip()
    label = (j.get("label") or "").strip().lower() or "primary care"
    radius = j.get("radius_km", 10)
    try:
        radius = float(radius)
    except Exception:
        radius = 10.0
    payload = search_providers(lat, lng, label, tax, radius_km=radius)
    # Strip precise coordinates (privacy) to coarse rounding
    payload["query_origin"] = {"lat": round(lat, 3), "lng": round(lng, 3)}
    return jsonify(payload)

# ---- High-quality TTS (Edge) for avatar lip-sync ----
@app.post("/api/tts")
def tts_edge():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing text"}), 400
    # strip references section so audio is clean
    idx = text.find("\n\nReferences")
    if idx >= 0:
        text = text[:idx].strip()

    ts = int(time.time())
    audio_path = os.path.join("/tmp/uploads", f"tts_{ts}.mp3")
    # Generate TTS using legacy edge-tts utility (faster than full video pipeline)
    try:
        import edge_tts
        async def gen():
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            await communicate.save(audio_path)
        asyncio.run(gen())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("TTS ERROR:", repr(e))
        return jsonify({"error": f"tts failed: {repr(e)}"}), 500

    # Return a static-served URL path
    # Expose uploads via a simple path; Flask will serve via send_file if needed
    return jsonify({
        "ok": True,
        "audio": f"/tmp/uploads/{os.path.basename(audio_path)}"
    })

# Generate a per-reply talking head video and matching audio
@app.post("/api/avatar")
def generate_avatar_video():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing text"}), 400
    # strip references section so audio/video are clean
    idx = text.find("\n\nReferences")
    if idx >= 0:
        text = text[:idx].strip()

    ts = int(time.time() * 1000)  # millisecond resolution for uniqueness
    audio_path = os.path.join("/tmp/uploads", f"av_{ts}.mp3")
    video_path = os.path.join("/tmp/uploads", f"av_{ts}.mp4")
    os.makedirs("/tmp/uploads", exist_ok=True)
    try:
        # This renders both audio and video; it will overwrite if exists
        asyncio.run(synth_and_render(text, audio_path, video_path))
    except Exception as e:
        import traceback
        print("AVATAR RENDER ERROR:")
        traceback.print_exc()

        try:
            import edge_tts
            async def gen():
                communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
                await communicate.save(audio_path)
            asyncio.run(gen())
        except Exception as tts_e:
            print("AVATAR FALLBACK TTS ERROR:", repr(tts_e))
            traceback.print_exc()
            return jsonify({"error": f"avatar render failed: {repr(e)}; tts fallback failed: {repr(tts_e)}"}), 500
        # Copy a static placeholder video to ensure a unique file exists
        try:
            import shutil
            static_vid = os.path.join(app.static_folder, "talking_head.mp4")
            if not os.path.exists(static_vid):
                return jsonify({"error": "static talking_head.mp4 missing"}), 500
            shutil.copyfile(static_vid, video_path)
        except Exception as copy_e:
            return jsonify({"error": f"avatar render failed: {e}; copy fallback failed: {copy_e}"}), 500

    return jsonify({
        "ok": True,
        "audio": f"/tmp/uploads/{os.path.basename(audio_path)}",
        "video": f"/tmp/uploads/{os.path.basename(video_path)}",
    })

# Serve files in uploads simply
@app.get('/tmp/uploads/<path:fname>')
def get_upload(fname):
    p = os.path.join('/tmp/uploads', fname)
    if not os.path.exists(p):
        return jsonify({"error": "not found"}), 404
    from flask import send_file
    return send_file(p, as_attachment=False)

# ---- Entrypoint ----
if __name__ == "__main__":
    # Disable reloader so in-memory histories aren't lost/reset across processes
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
