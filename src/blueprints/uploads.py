import os
import asyncio
from flask import Blueprint, request, jsonify
from src.utils.files import safe_save, ensure_dir
from src.services.genomics import run_genomics
from src.services.avatar import synth_and_render

bp = Blueprint('uploads', __name__)
BASE_UPLOAD = os.path.abspath('uploads')
ensure_dir(BASE_UPLOAD)

@bp.post('/genomic')
def upload_genomic():
    f = request.files.get('genomicFile')
    if not f or not f.filename: return jsonify({"error": "No file"}), 400
    saved = safe_save(f, BASE_UPLOAD, 'genomics')
    out_csv = os.path.join(BASE_UPLOAD, 'gene_conditions.csv')
    records = run_genomics(saved, out_csv)
    # Optionally push a summary into the chat memory later
    return jsonify({"ok": True, "records": records})

@bp.post('/avatar')
def make_avatar():
    text = request.form.get('text','').strip()
    if not text: return jsonify({"error": "Missing text"}), 400
    audio_path = os.path.join('static', 'tts_audio.mp3')
    video_path = os.path.join('static', 'talking_head.mp4')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(synth_and_render(text, audio_path, video_path))
    return jsonify({"ok": True, "video": video_path})