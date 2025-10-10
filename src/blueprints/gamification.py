import os, sqlite3, time
from flask import Blueprint, request, jsonify

bp = Blueprint('gami', __name__)
DB = os.path.abspath('hms_gami.sqlite')

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  type TEXT,
  ts INTEGER
);
CREATE TABLE IF NOT EXISTS rewards(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  label TEXT,
  ts INTEGER
);
"""

with sqlite3.connect(DB) as cx:
    cx.executescript(SCHEMA)

RULES = {
  'survey.completed': 'First Check-in',
  'streak.3': 'On a Roll',
  'streak.7': 'One-Week Streak',
}

@bp.post('/event')
def log_event():
    data = request.get_json(force=True)
    sid = data.get('session_id'); etype = data.get('type')
    if not sid or not etype: return jsonify({"error":"missing"}), 400
    now = int(time.time())
    with sqlite3.connect(DB) as cx:
        cx.execute("INSERT INTO events(session_id,type,ts) VALUES(?,?,?)", (sid, etype, now))
        cur = cx.execute("SELECT COUNT(*) FROM events WHERE session_id=? AND type='daily.login' AND ts> ?", (sid, now-86400*7))
        streak = cur.fetchone()[0]
        label = RULES.get(f'streak.{streak}') or RULES.get(etype)
        if label:
            cx.execute("INSERT INTO rewards(session_id,label,ts) VALUES(?,?,?)", (sid,label,now))
    return jsonify({"ok": True})

@bp.get('/rewards')
def get_rewards():
    sid = request.args.get('session_id','')
    with sqlite3.connect(DB) as cx:
        rows = cx.execute("SELECT label,ts FROM rewards WHERE session_id=? ORDER BY ts DESC", (sid,)).fetchall()
    return jsonify([{"label": r[0], "ts": r[1]} for r in rows])
