import eng_to_ipa as ipa
import moviepy.editor as mp
from moviepy.video.VideoClip import ImageClip
import os
from pydub import AudioSegment
from pydub.silence import detect_silence
import random
import edge_tts
import asyncio
from typing import Dict, List, Tuple, Optional
import time
import subprocess

# NOTE: This module is performance tuned (fast path) to reduce the time spent
# inside moviepy/ffmpeg when stitching MANY tiny clips. Key changes:
# 1. Global cache of preloaded viseme clips (avoid re-decoding each run)
# 2. Silence aggregation (one clip per contiguous silent region instead of 0.1s pieces)
# 3. Use method='chain' instead of 'compose' when possible (no costly compositing)
# 4. Minimize number of concatenated segments (major I/O + encoding savings)
# 5. Optional FAST mode via AVATAR_FAST=1 (default on) with simplified timing logic
# 6. Allow overriding output bitrate / crf for faster encode

FAST_MODE = os.getenv("AVATAR_FAST", "1") != "0"
DEFAULT_FPS = 20
DEFAULT_CODEC = "libx264"
DEFAULT_PRESET = os.getenv("AVATAR_FFMPEG_PRESET", "ultrafast")
DEFAULT_CRF = os.getenv("AVATAR_CRF", "32")  # higher CRF => lower quality + faster
DEFAULT_BITRATE = os.getenv("AVATAR_BITRATE")  # optional, e.g., "800k"

# Global cache to avoid reloading video clips every invocation
_VISEME_CACHE: Dict[Tuple[str, Tuple[str, ...]], Dict[str, mp.VideoFileClip]] = {}
# Track original file paths to enable zero-reencode concat mode
_VISEME_PATHS_CACHE: Dict[Tuple[str, Tuple[str, ...]], Dict[str, str]] = {}
COPY_MODE = os.getenv("AVATAR_COPY_MODE", "1") != "0"  # attempt concat demuxer copy
FORCE_REENCODE = os.getenv("AVATAR_FORCE_REENCODE", "0") == "1"
HYBRID_ENABLED = os.getenv("AVATAR_HYBRID_CONCAT", "1") != "0"  # try fast single-pass re-encode before Python assembly
CONCAT_REENCODE = os.getenv("AVATAR_CONCAT_REENCODE", "1") != "0"  # ffmpeg concat + re-encode (faster than Python/moviepy)
SCALE_FILTER = os.getenv("AVATAR_SCALE")  # e.g. "640:-2" to downscale for speed
NORMALIZE_ENABLED = os.getenv("AVATAR_NORMALIZE", "1") != "0"  # build uniform intermediate set for safe copy-mode
TARGET_FPS = int(os.getenv("AVATAR_TARGET_FPS", str(DEFAULT_FPS)))
NORMALIZED_SUBDIR = os.getenv("AVATAR_NORMALIZED_DIR", "_normalized")
DEBUG = os.getenv("AVATAR_DEBUG", "0") == "1"
SYNC_MODE = os.getenv("AVATAR_SYNC", "0") == "1"  # timing-aware assembly; skips copy-mode fast heuristics
MIN_BASE_VIS_DUR = float(os.getenv("AVATAR_MIN_VIS_DUR", "0.5"))  # ensure normalized viseme clips aren't too tiny
SILENCE_MIN_LEN = int(os.getenv("AVATAR_SILENCE_MIN_LEN", "10"))   # ms
SILENCE_THRESH = int(os.getenv("AVATAR_SILENCE_THRESH", "-40"))     # dBFS threshold
USE_IMAGE_CLIPS = os.getenv("AVATAR_USE_IMAGE_CLIPS", "0") == "1"  # force freezing viseme videos into single-frame ImageClips
SPEAKING_FRACTION = float(os.getenv("AVATAR_SPEAKING_FRACTION", "0.75"))  # portion of speech interval actually showing mouth movement
IDLE_MIN_GAP = float(os.getenv("AVATAR_IDLE_MIN_GAP", "0.05"))  # minimum neutral gap inserted
IDLE_MAX_GAP = float(os.getenv("AVATAR_IDLE_MAX_GAP", "0.35"))  # cap for any single idle gap

# Mapping of phonemes to visemes
phoneme_to_viseme = {
    'b': 'p', 'd': 't', 'ʤ': 'S_cap', 'ð': 'T_cap', 'f': 'f', 'ɡ': 'k', 'h': 'k', 'j': 'i', 'k': 'k',
    'l': 'l', 'm': 'p', 'n': 't', 'ŋ': 'k', 'p': 'p', 'r': 'r', 's': 's', 'ʃ': 'S_cap', 't': 't',
    'ʧ': 'S_cap', 'θ': 'T_cap', 'v': 'f', 'w': 'u', 'z': 's', 'ʒ': 'S_cap', 'ə': '@', 'ər': '@',
    'æ': 'a', 'aɪ': 'a', 'aʊ': 'a', 'ɑ': 'a', 'eɪ': 'e', 'ɛ': 'e', 'i': 'i', 'ɪ': 'i', 'oʊ': 'o',
    'ɔ': 'O_cap', 'ɔɪ': 'O_cap', 'u': 'u', 'ʊ': 'u', ' ': 'default', ',': 'default', '.': 'default',
}

def text_to_phonemes(text):
    """
    Convert text to phonemes using the eng_to_ipa library.

    Args:
        text (str): The input text to convert.

    Returns:
        str: The converted phonemes.
    """
    phonemes = ipa.convert(text)
    return phonemes

def text_to_visemes(text):
    """
    Convert text to visemes using the phoneme_to_viseme mapping.

    Args:
        text (str): The input text to convert.

    Returns:
        list: A list of visemes corresponding to the input text.
    """
    phonemes = text_to_phonemes(text)
    visemes = []
    for p in phonemes:
        chance = random.random()
        if p in phoneme_to_viseme and chance <= 0.8: # randomly skip some visemes to reduce video length
            visemes.append(phoneme_to_viseme[p])
    return visemes

async def text_to_speech(text, output_path, voice="en-US-GuyNeural"):
    """
    Generate TTS audio file from text using the Edge TTS API.

    Args:
        text (str): The input text to convert to speech.
        output_path (str): The path to save the generated audio file.
        voice (str, optional): The voice to use for TTS. Defaults to "en-US-GuyNeural".
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def detect_silence_segments(audio_path, min_silence_len=None, silence_thresh=None):
    """
    Detect silent segments in a TTS audio file.

    Args:
        audio_path (str): The path to the audio file.
        min_silence_len (int, optional): Minimum length of silence to detect (in ms). Defaults to 200.
        silence_thresh (int, optional): Silence threshold (in dB). Defaults to -40.

    Returns:
        list: A list of tuples representing the start and end of silent segments.
    """
    audio = AudioSegment.from_file(audio_path)
    if min_silence_len is None:
        min_silence_len = SILENCE_MIN_LEN
    if silence_thresh is None:
        silence_thresh = SILENCE_THRESH
    silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return silence_ranges

def preload_viseme_clips(viseme_list: List[str], base_path: str) -> Dict[str, mp.VideoFileClip]:
    """Legacy full preload (kept for compatibility) -- now generally avoided.

    Still used if caller explicitly wants every clip resident; otherwise we
    lazily load only required visemes via `_lazy_load_required_visemes`.
    """
    base_path = os.path.abspath(base_path)
    key = (base_path, tuple(sorted(viseme_list)))
    cached = _VISEME_CACHE.get(key)
    if cached:
        return cached
    t0 = time.time()
    clips: Dict[str, mp.VideoFileClip] = {}
    paths: Dict[str, str] = {}
    for viseme in viseme_list:
        clip_path = os.path.join(base_path, f"{viseme}.mp4")
        if not os.path.exists(clip_path):
            continue
        try:
            # audio=False shaves probe time
            clip = mp.VideoFileClip(clip_path, audio=False)
            clips[viseme] = clip
            paths[viseme] = clip_path
        except Exception as e:
            print(f"[avatar] preload failure {clip_path}: {e}")
    default_path = os.path.join(base_path, "default.mp4")
    if os.path.exists(default_path):
        try:
            clips['default'] = mp.VideoFileClip(default_path, audio=False)
            paths['default'] = default_path
        except Exception as e:
            print(f"[avatar] Failed to load default clip: {e}")
    _VISEME_CACHE[key] = clips
    _VISEME_PATHS_CACHE[key] = paths
    print(f"[avatar] Full preload loaded {len(clips)} clips in {time.time()-t0:.3f}s")
    return clips


def _lazy_load_required_visemes(required: List[str], base_path: str) -> Dict[str, mp.VideoFileClip]:
    """Load only the subset of viseme clips actually needed for this utterance.

    Caches each clip individually (shared across different required sets) so future
    loads are O(1) dictionary hits. This avoids paying cost for 15+ ffprobe runs
    when only (for example) 5 visemes appear in a short sentence.
    """
    base_path = os.path.abspath(base_path)
    # Use a wide key (base_path + *all* canonical visemes) for cache grouping; but allow per-clip caching.
    # We'll store a special key for aggregated set; the mutable cache dict will accumulate members.
    aggregate_key = (base_path, tuple(sorted(["__aggregate_lazy_cache__"])))
    store = _VISEME_CACHE.setdefault(aggregate_key, {})
    t0 = time.time(); loaded_now = 0
    def _verify_clip_readable(vclip: mp.VideoFileClip, path: str) -> bool:
        try:
            # Try grabbing first frame; some corrupt clips load metadata but fail on frame decode
            vclip.get_frame(0)
            return True
        except Exception as e:
            if DEBUG:
                print(f"[avatar] Frame decode failed ({path}): {e}")
            return False

    def _attempt_repair(path: str, fallback: Optional[str]) -> Optional[mp.VideoFileClip]:
        """Re-encode problematic viseme to a safe h264 clip. Returns VideoFileClip or None."""
        try:
            repaired = path + ".repair.mp4"
            cmd = [
                "ffmpeg","-y","-i",path,
                "-vf",f"fps={TARGET_FPS}",
                "-c:v","libx264","-preset", DEFAULT_PRESET,"-crf", DEFAULT_CRF,
                "-pix_fmt","yuv420p","-movflags","+faststart","-an", repaired
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if os.path.exists(repaired) and os.path.getsize(repaired) > 800:
                clip = mp.VideoFileClip(repaired, audio=False)
                if _verify_clip_readable(clip, repaired):
                    if DEBUG:
                        print(f"[avatar] Repaired viseme clip {os.path.basename(path)} -> {os.path.basename(repaired)}")
                    return clip
        except Exception as e:
            if DEBUG:
                print(f"[avatar] repair failed {path}: {e}")
        # fallback duplication
        if fallback and os.path.exists(fallback):
            try:
                fclip = mp.VideoFileClip(fallback, audio=False)
                if _verify_clip_readable(fclip, fallback):
                    return fclip
            except Exception:
                return None
        return None

    default_candidate = os.path.join(base_path, 'default.mp4')
    freeze = USE_IMAGE_CLIPS or SYNC_MODE
    for vis in required:
        if vis in store:
            continue
        clip_path = os.path.join(base_path, f"{vis}.mp4")
        if not os.path.exists(clip_path):
            continue
        try:
            clip = mp.VideoFileClip(clip_path, audio=False)
            if not _verify_clip_readable(clip, clip_path):
                # try repair
                clip.close()
                repaired = _attempt_repair(clip_path, default_candidate)
                if repaired:
                    store[vis] = repaired
                    loaded_now += 1
                else:
                    if DEBUG:
                        print(f"[avatar] Could not repair {vis}; using default mapping")
                    # map to default later
                continue
            if freeze:
                try:
                    frame0 = clip.get_frame(0)
                    # Choose a representative duration for raw clip when freezing; use clip.duration capped between MIN_BASE_VIS_DUR and 0.6
                    base_dur = clip.duration or MIN_BASE_VIS_DUR
                    base_dur = max(MIN_BASE_VIS_DUR, min(0.6, base_dur))
                    iclip = ImageClip(frame0).set_duration(base_dur)
                    store[vis] = iclip
                    # Close original to free resources
                    try: clip.close()
                    except Exception: pass
                except Exception as e:
                    if DEBUG:
                        print(f"[avatar] freeze conversion failed {clip_path}: {e}")
                    store[vis] = clip  # fallback to original
            else:
                store[vis] = clip
            loaded_now += 1
        except Exception as e:
            if DEBUG:
                print(f"[avatar] lazy load fail {clip_path}: {e}")
            repaired = _attempt_repair(clip_path, default_candidate)
            if repaired:
                store[vis] = repaired
                loaded_now += 1
    if loaded_now:
        print(f"[avatar] Lazy loaded {loaded_now} new viseme clips in {time.time()-t0:.3f}s (total cached={len(store)})")
    # Always ensure default present if exists
    if 'default' not in store:
        dpath = os.path.join(base_path, 'default.mp4')
        if os.path.exists(dpath):
            try:
                store['default'] = mp.VideoFileClip(dpath, audio=False)
            except Exception as e:
                print(f"[avatar] lazy load default fail: {e}")
    return store


def _aggregate_silence(silence_ranges: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Merge silence ranges into a simplified list (ms)."""
    if not silence_ranges:
        return []
    silence_ranges = sorted(silence_ranges)
    merged = [silence_ranges[0]]
    for start, end in silence_ranges[1:]:
        ls, le = merged[-1]
        if start <= le + 50:  # merge close/overlapping (<=50ms gap)
            merged[-1] = (ls, max(le, end))
        else:
            merged.append((start, end))
    return [(s/1000.0, e/1000.0) for s, e in merged]


def create_video_from_visemes(visemes: List[str], silence_ranges: List[Tuple[int, int]], audio_duration: float, viseme_clips: Dict[str, mp.VideoFileClip]):
    """Construct video timeline from visemes with aggregated silence intervals.

    Fast path reduces the number of concatenated segments dramatically.
    """
    if not viseme_clips:
        raise ValueError("No viseme clips loaded.")

    # Aggregate silence to fewer segments
    silence_intervals = _aggregate_silence(silence_ranges)

    # Convert to a quick lookup to test whether a time point is inside a silence interval
    # We'll iterate through visemes sequentially while tracking time
    def is_silence(t: float) -> bool:
        for s, e in silence_intervals:
            if s <= t < e:
                return True
        return False

    clips = []
    current_time = 0.0
    v_index = 0
    default_clip = viseme_clips.get('default')
    # Guard: if no default clip, fallback to first available
    if default_clip is None:
        default_clip = next(iter(viseme_clips.values()))

    # We'll build larger silence chunks instead of 0.1s pieces
    while current_time < audio_duration and (v_index < len(visemes) or is_silence(current_time)):
        if is_silence(current_time):
            # find end of this silence block
            end_block = current_time
            for s, e in silence_intervals:
                if s <= current_time < e:
                    end_block = e
                    break
            dur = max(0.05, min(end_block - current_time, audio_duration - current_time))
            clips.append(default_clip.set_duration(dur))
            current_time += dur
            continue
        # speech region
        if v_index < len(visemes):
            vis = visemes[v_index]
            base = viseme_clips.get(vis) or default_clip
            clips.append(base)
            current_time += base.duration
            v_index += 1
        else:
            # trailing area with no visemes, fill with default
            remaining = audio_duration - current_time
            if remaining <= 0: break
            clips.append(default_clip.set_duration(remaining))
            break

    if not clips:
        return None
    # Use 'chain' (faster) if all clips share same size / FPS (MoviePy will check)
    method = "chain"
    try:
        sizes = {c.size for c in clips if hasattr(c, 'size')}
        if len(sizes) > 1:
            method = "compose"
    except Exception:
        method = "compose"
    return mp.concatenate_videoclips(clips, method=method)


def _build_timed_video(visemes: List[str], silence_ranges_ms: List[Tuple[int,int]], audio_duration: float, viseme_clips: Dict[str, mp.VideoFileClip]):
    """Timing-aware assembly: allocate viseme durations to fit speech regions and explicit silence blocks.

    Strategy:
      1. Derive merged silence intervals (seconds) inside [0, audio_duration].
      2. Compute speech intervals between them.
      3. Allocate number of visemes to each speech interval proportional to its duration.
      4. Inside each speech interval, assign uniform target duration per viseme; trim or stretch (via set_duration) base clips.
      5. Silence intervals represented by a neutral 'default' clip with exact silence duration.

    This yields a deterministic mapping ensuring final concatenation length matches (or is extremely close to) audio duration.
    """
    if not visemes:
        return None
    default_clip = viseme_clips.get('default') or next(iter(viseme_clips.values()))
    # Merge & clamp silence
    sil = _aggregate_silence(silence_ranges_ms)
    sil = [(max(0.0,s), min(audio_duration,e)) for s,e in sil if s < audio_duration]
    # Build speech intervals
    speech_intervals = []
    cursor = 0.0
    for s,e in sil:
        if cursor < s:
            speech_intervals.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < audio_duration:
        speech_intervals.append((cursor, audio_duration))
    total_speech = sum(e - s for s,e in speech_intervals)
    if total_speech <= 0:
        # All silence
        return mp.concatenate_videoclips([default_clip.set_duration(audio_duration)], method='chain')
    # Allocate viseme counts per speech interval
    remaining_visemes = len(visemes)
    allocation = []  # (interval, count)
    for idx,(s,e) in enumerate(speech_intervals):
        interval_len = e - s
        if idx == len(speech_intervals) - 1:
            count = remaining_visemes  # assign remainder
        else:
            # proportional, at least 1 if any visemes remain
            ideal = (interval_len / total_speech) * len(visemes)
            count = max(1, int(round(ideal))) if remaining_visemes > 0 else 0
            if count > remaining_visemes:
                count = remaining_visemes
        allocation.append(((s,e), count))
        remaining_visemes -= count
    # If we over-allocated due to rounding, trim from largest intervals
    if remaining_visemes < 0:
        over = -remaining_visemes
        for i in sorted(range(len(allocation)), key=lambda i: allocation[i][0][1]-allocation[i][0][0], reverse=True):
            if over <= 0: break
            interval,count = allocation[i]
            if count > 1:
                allocation[i] = (interval, count-1)
                over -= 1
    # Build sequence
    viseme_iter = iter(visemes)
    segments = []
    # Interleave: [speech block visemes] + following silence (if any)
    sil_idx = 0
    def _append_silence(until_start, until_end):
        dur = max(0.0, until_end - until_start)
        if dur > 0.001:
            segments.append(default_clip.set_duration(dur))
    # We'll recreate order by scanning timeline
    timeline_points = []
    for interval,count in allocation:
        timeline_points.append(('speech', interval, count))
    for s,e in sil:
        timeline_points.append(('sil', (s,e), 0))
    # Sort by start time
    timeline_points.sort(key=lambda x: x[1][0])
    # Iterate maintaining which visemes consumed
    for kind,(s,e),count in timeline_points:
        if kind == 'sil':
            _append_silence(s,e)
        else:
            interval_len = max(0.0, e - s)
            if count <= 0 or interval_len <= 0:
                continue
            # Decide how much of this interval is active speaking vs idle neutral
            speaking_target = interval_len * min(0.95, max(0.05, SPEAKING_FRACTION))
            # Initial base duration per viseme
            MIN_DUR = 0.05
            MAX_DUR = 0.55
            base_td = speaking_target / count
            base_td = min(MAX_DUR, max(MIN_DUR, base_td))
            total_active = base_td * count
            idle_total = max(0.0, interval_len - total_active)
            # Distribute idle across boundaries (count+1 gaps)
            gaps = count + 1
            if idle_total > 0 and gaps > 0:
                raw_gap = idle_total / gaps
                gap_val = min(IDLE_MAX_GAP, max(0.0, raw_gap))
            else:
                gap_val = 0.0
            # If gap too small to be meaningful (< IDLE_MIN_GAP) collapse into viseme durations
            if gap_val < IDLE_MIN_GAP and idle_total > 0:
                # absorb idle into active speaking evenly
                extra = idle_total / count
                base_td = min(MAX_DUR, base_td + extra)
                gap_val = 0.0
            if DEBUG:
                print(f"[avatar-sync] interval {s:.2f}-{e:.2f}s len={interval_len:.2f}s vis={count} base_td={base_td:.3f}s gap={gap_val:.3f}s idle_total={idle_total:.2f}s")
            # Build: leading idle gap
            if gap_val > 0:
                segments.append(default_clip.set_duration(gap_val))
            for i in range(count):
                try:
                    v = next(viseme_iter)
                except StopIteration:
                    v = 'default'
                base = viseme_clips.get(v) or default_clip
                td = base_td
                try:
                    if hasattr(base, 'duration') and base.duration > td:
                        clip = base.subclip(0, td)
                    else:
                        clip = base.set_duration(td)
                except Exception:
                    clip = base.set_duration(td)
                segments.append(clip)
                # trailing idle gap after each viseme (except last if we already covered remainder)
                if gap_val > 0:
                    segments.append(default_clip.set_duration(gap_val))
    if not segments:
        return default_clip.set_duration(audio_duration)
    # Concatenate (chain if possible)
    try:
        sizes = {c.size for c in segments if hasattr(c,'size')}
        method = 'chain' if len(sizes) == 1 else 'compose'
    except Exception:
        method = 'compose'
    timeline = mp.concatenate_videoclips(segments, method=method)
    # Adjust final timeline length slight drift to match audio_duration
    drift = audio_duration - timeline.duration
    if abs(drift) > 0.15:  # only correct if significant
        if DEBUG:
            print(f"[avatar] Timing drift {drift:+.2f}s -> adjusting final duration")
        if drift > 0:
            timeline = mp.concatenate_videoclips([timeline, default_clip.set_duration(drift)], method='chain')
        else:
            try:
                timeline = timeline.subclip(0, audio_duration)
            except Exception:
                pass
    return timeline


def _can_use_copy_mode(visemes: List[str], base_path: str, viseme_list: List[str]) -> bool:
    """Check if we have all needed files to use ffmpeg concat demuxer with -c copy.

    Copy mode constraints:
      - All referenced viseme names must have a backing .mp4 file (including default)
      - We will not alter durations; we simply sequence each clip once per occurrence
        (silence intervals are represented by the default clip repeated). If we
        need fractional trimming, copy mode is skipped.
    """
    if not COPY_MODE or FORCE_REENCODE:
        return False
    base_path = os.path.abspath(base_path)
    required = set([v for v in visemes if v in viseme_list]) | {"default"}
    for v in required:
        if not os.path.exists(os.path.join(base_path, f"{v}.mp4")):
            return False
    return True


def _probe_stream_info(path: str) -> Optional[dict]:
    try:
        import json
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,pix_fmt",
            "-of", "json", path
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        js = json.loads(out.stdout or '{}')
        streams = js.get('streams') or []
        if not streams:
            return None
        st = streams[0]
        # derive numeric fps
        def _fps(val: str) -> float:
            if not val or '/' not in val:
                return 0.0
            n,d = val.split('/')
            try:
                return float(n)/float(d)
            except Exception:
                return 0.0
        st['fps'] = _fps(st.get('r_frame_rate') or st.get('avg_frame_rate'))
        return st
    except Exception as e:
        if DEBUG:
            print(f"[avatar] stream probe failed {path}: {e}")
        return None


def _needs_normalization(base_path: str, viseme_list: List[str]) -> bool:
    """Heuristic: if any clip differs in codec/pix_fmt/fps/size from first reference -> normalize.

    Also if codec != libx264 or pix_fmt != yuv420p or fps != TARGET_FPS."""
    infos = []
    ref = None
    for v in viseme_list:
        p = os.path.join(base_path, f"{v}.mp4")
        if not os.path.exists(p):
            continue
        info = _probe_stream_info(p)
        if not info:
            return True
        infos.append(info)
        if ref is None:
            ref = info
            continue
        for key in ("codec_name","width","height","pix_fmt"):
            if info.get(key) != ref.get(key):
                return True
        # allow small fps drift (<0.5)
        if abs(info.get('fps',0)-ref.get('fps',0)) > 0.5:
            return True
    # Additional strictness for copy-mode success
    for info in infos:
        if info.get('codec_name') != 'h264':
            return True
        if info.get('pix_fmt') not in ('yuv420p','yuvj420p'):
            return True
        # unify fps with target
        if abs(info.get('fps',0) - TARGET_FPS) > 0.5:
            return True
    return False


def _ensure_normalized_clips(base_path: str, viseme_list: List[str]) -> str:
    """Create a normalized set (libx264, yuv420p, TARGET_FPS, uniform size) for safe copy-mode.

    Returns path to directory containing normalized clips (may be the original if already uniform)."""
    base_path = os.path.abspath(base_path)
    if not NORMALIZE_ENABLED:
        return base_path
    normalize_required = _needs_normalization(base_path, viseme_list)
    norm_dir = os.path.join(base_path, NORMALIZED_SUBDIR)
    if not normalize_required and all(os.path.exists(os.path.join(base_path, f"{v}.mp4")) for v in viseme_list):
        if DEBUG:
            print("[avatar] Normalization not required; using original clips")
        return base_path
    os.makedirs(norm_dir, exist_ok=True)
    # Determine reference size (use default.mp4 if present else first existing)
    ref_w = ref_h = None
    for v in (['default'] + viseme_list):
        p = os.path.join(base_path, f"{v}.mp4")
        if os.path.exists(p):
            info = _probe_stream_info(p)
            if info:
                ref_w, ref_h = info.get('width'), info.get('height')
                break
    # Fallback size
    if not ref_w:
        ref_w, ref_h = 512, 720
    # Build each normalized file if missing or older than source
    for v in viseme_list:
        src = os.path.join(base_path, f"{v}.mp4")
        if not os.path.exists(src):
            continue
        dst = os.path.join(norm_dir, f"{v}.mp4")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        scale_expr = f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease"
        vf_chain = [scale_expr, f"fps={TARGET_FPS}"]
        if SCALE_FILTER:  # if user explicitly wants additional scale override
            vf_chain.insert(0, f"scale={SCALE_FILTER}")
        cmd = [
            "ffmpeg","-y","-i",src,
            "-vf"," ,".join(vf_chain),
            "-c:v","libx264","-preset", DEFAULT_PRESET,"-crf", DEFAULT_CRF,
            "-pix_fmt","yuv420p","-movflags","+faststart","-an", dst
        ]
        if DEBUG:
            print("[avatar] Normalizing", src, "->", dst)
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Post-normalization duration padding if clip extremely short
        try:
            dur = _probe_duration(dst)
            if 0 < dur < MIN_BASE_VIS_DUR:
                pad = max(0.0, MIN_BASE_VIS_DUR - dur)
                if pad > 0.01:
                    padded = dst + ".pad.tmp.mp4"
                    # Use tpad to clone last frame; maintain fps & size
                    vf_pad = f"fps={TARGET_FPS},tpad=stop_mode=clone:stop_duration={pad:.3f}"
                    if SCALE_FILTER:
                        vf_pad = f"scale={SCALE_FILTER},{vf_pad}"
                    subprocess.run([
                        "ffmpeg","-y","-i",dst,"-vf",vf_pad,
                        "-c:v","libx264","-preset", DEFAULT_PRESET,"-crf", DEFAULT_CRF,
                        "-pix_fmt","yuv420p","-movflags","+faststart","-an", padded
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(padded) and os.path.getsize(padded) > 500:
                        os.replace(padded, dst)
                        if DEBUG:
                            print(f"[avatar] Padded viseme {v} from {dur:.2f}s to >= {MIN_BASE_VIS_DUR:.2f}s")
        except Exception as e:
            if DEBUG:
                print(f"[avatar] pad check failed for {dst}: {e}")
    # Quick verification; if too few produced, fallback to original path (will later re-encode)
    produced = sum(1 for v in viseme_list if os.path.exists(os.path.join(norm_dir, f"{v}.mp4")))
    if produced < 3:
        if DEBUG:
            print(f"[avatar] Normalization produced only {produced} clips; ignoring normalized set")
        return base_path
    if DEBUG:
        print(f"[avatar] Using normalized viseme directory ({produced} clips) -> {norm_dir}")
    return norm_dir


def _build_copy_mode_video(visemes: List[str], base_path: str, temp_video: str, audio_path: str, final_output_path: str) -> bool:
    """Attempt ultra-fast assembly using ffmpeg concat demuxer + audio remux.

    Returns True on success else False (caller can fallback).
    NOTE: This approach ignores precise silence timing; it sequences one viseme clip
          per viseme symbol and inserts a default clip for runs of 'silence' markers.
    """
    try:
        list_path = temp_video + ".list"
        base_path = os.path.abspath(base_path)
        # Collapse consecutive identical visemes (including default) to reduce segments
        collapsed: List[str] = []
        last = None
        for v in visemes:
            if v != last:
                collapsed.append(v)
                last = v
        if not collapsed:
            collapsed = ["default"]
        # Pad sequence with default clips to roughly match audio duration (if audio available)
        audio_dur = 0.0
        try:
            import json
            probe = subprocess.run([
                "ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=duration","-of","json", audio_path
            ], capture_output=True, text=True, timeout=5)
            if probe.returncode == 0:
                js = json.loads(probe.stdout or '{}')
                streams = js.get('streams') or []
                if streams and 'duration' in streams[0]:
                    audio_dur = float(streams[0]['duration'])
        except Exception:
            audio_dur = 0.0
        # Estimate duration of collapsed visemes; approximate by summing clip durations
        def _seq_duration(seq):
            return sum(_probe_duration(os.path.join(base_path, f"{v}.mp4")) for v in seq)
        total_viseme_dur = _seq_duration(collapsed)
        if audio_dur > 0 and total_viseme_dur > 0:
            # Repeat the collapsed sequence until reaching ~90% of audio, then pad with default
            if total_viseme_dur < audio_dur * 0.9:
                repeats = int((audio_dur * 0.9) / max(total_viseme_dur, 0.01)) - 1
                repeats = max(0, min(repeats, 20))
                if repeats:
                    if DEBUG:
                        print(f"[avatar] Repeating viseme block {repeats} times to extend duration")
                    collapsed = collapsed * (repeats + 1)
                total_viseme_dur = _seq_duration(collapsed)
            if total_viseme_dur < audio_dur * 0.85:
                default_path = os.path.join(base_path, "default.mp4")
                def_dur = _probe_duration(default_path) or 0.25
                needed = int(((audio_dur - total_viseme_dur) / def_dur) + 0.5)
                needed = min(needed, 1000)
                if DEBUG:
                    print(f"[avatar] Padding copy-mode with {needed} default clips after repeats (have {total_viseme_dur:.2f}s need {audio_dur:.2f}s)")
                collapsed.extend(["default"] * needed)
        if audio_dur > 0 and _seq_duration(collapsed) < audio_dur * 0.5:
            if DEBUG:
                print(f"[avatar] Copy-mode preflight duration too short ({_seq_duration(collapsed):.2f}s vs audio {audio_dur:.2f}s); aborting copy-mode")
            return False
        with open(list_path, "w") as f:
            for v in collapsed:
                path = os.path.join(base_path, f"{v}.mp4")
                f.write(f"file '{path}'\n")
        # --- Attempt pure copy mode first (fastest) ---
        video_only = temp_video + "_video.mp4"
        cmd_copy = [
            "ffmpeg","-y","-f","concat","-safe","0","-i",list_path,
            "-fflags","+genpts","-c","copy","-movflags","+faststart", video_only
        ]
        t0 = time.time()
        copy_ok = True
        try:
            if DEBUG:
                proc = subprocess.run(cmd_copy, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"[avatar] Pure copy concat stderr:\n{proc.stderr[:4000]}")
                    raise subprocess.CalledProcessError(proc.returncode, cmd_copy, proc.stdout, proc.stderr)
            else:
                subprocess.run(cmd_copy, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[avatar] Pure copy concat failed: {e}")
            copy_ok = False

        if copy_ok:
            cmd_mux = [
                "ffmpeg","-y","-i",video_only,"-i",audio_path,
                "-c:v","copy","-c:a","aac","-shortest","-movflags","+faststart", final_output_path
            ]
            try:
                if DEBUG:
                    proc2 = subprocess.run(cmd_mux, capture_output=True, text=True)
                    if proc2.returncode != 0:
                        print(f"[avatar] Copy-mode mux stderr:\n{proc2.stderr[:4000]}")
                        raise subprocess.CalledProcessError(proc2.returncode, cmd_mux, proc2.stdout, proc2.stderr)
                else:
                    subprocess.run(cmd_mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                dt = time.time()-t0
                print(f"[avatar] Copy-mode assembly succeeded in {dt:.2f}s (segments={len(collapsed)})")
                return True
            except Exception as e:
                print(f"[avatar] Copy-mode mux failed: {e}")

        # --- Hybrid fast re-encode (still much faster than Python path) ---
        if HYBRID_ENABLED:
            print("[avatar] Falling back to hybrid fast concat re-encode...")
            video_hybrid = temp_video + "_hybrid.mp4"
            cmd_hybrid = [
                "ffmpeg","-y","-f","concat","-safe","0","-i",list_path,
                "-c:v","libx264","-preset", DEFAULT_PRESET, "-crf", DEFAULT_CRF,
                "-pix_fmt","yuv420p","-movflags","+faststart", video_hybrid
            ]
            try:
                if DEBUG:
                    proc3 = subprocess.run(cmd_hybrid, capture_output=True, text=True)
                    if proc3.returncode != 0:
                        print(f"[avatar] Hybrid concat stderr:\n{proc3.stderr[:4000]}")
                        raise subprocess.CalledProcessError(proc3.returncode, cmd_hybrid, proc3.stdout, proc3.stderr)
                else:
                    subprocess.run(cmd_hybrid, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Mux audio (avoid re-encoding video again)
                cmd_mux2 = [
                    "ffmpeg","-y","-i",video_hybrid,"-i",audio_path,
                    "-c:v","copy","-c:a","aac","-shortest","-movflags","+faststart", final_output_path
                ]
                if DEBUG:
                    proc4 = subprocess.run(cmd_mux2, capture_output=True, text=True)
                    if proc4.returncode != 0:
                        print(f"[avatar] Hybrid mux stderr:\n{proc4.stderr[:4000]}")
                        raise subprocess.CalledProcessError(proc4.returncode, cmd_mux2, proc4.stdout, proc4.stderr)
                else:
                    subprocess.run(cmd_mux2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                dt = time.time()-t0
                print(f"[avatar] Hybrid concat succeeded in {dt:.2f}s (segments={len(collapsed)})")
                return True
            except Exception as e:
                print(f"[avatar] Hybrid concat failed: {e}")
        # If we reach here, signal failure so Python assembly can proceed
        return False
    except Exception as e:
        print(f"[avatar] Copy-mode failed: {e}")
        return False
def _validate_media(audio_path: str, video_path: str, expected_audio_sec: float) -> bool:
    """Validate the produced media has non-trivial duration and (roughly) matches audio.

    Returns True if acceptable, else False.
    """
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 500:
            print("[avatar] Audio file missing or too small")
            return False
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 800:
            print("[avatar] Video file missing or too small")
            return False
        # Lightweight duration check using moviepy (ffprobe under hood)
        vdur = mp.VideoFileClip(video_path).duration
        if not vdur or vdur < 0.2:
            print(f"[avatar] Video duration invalid: {vdur}")
            return False
        # If copy-mode produced something much shorter than audio and >1s audio expected, reject
        if expected_audio_sec > 2 and vdur < expected_audio_sec * 0.4:
            print(f"[avatar] Video too short vs audio (video={vdur:.2f}s audio={expected_audio_sec:.2f}s) -> fallback")
            return False
        return True
    except Exception as e:
        print(f"[avatar] Validation exception: {e}")
        return False


# --------- FFmpeg concat re-encode path (without MoviePy heavy object creation) ---------
_DURATION_CACHE: Dict[str, float] = {}

def _probe_duration(path: str) -> float:
    if path in _DURATION_CACHE:
        return _DURATION_CACHE[path]
    try:
        # Use ffprobe for speed; fallback to moviepy if missing
        import json, subprocess
        cmd = ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=duration","-of","json", path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        dur = 0.0
        if out.returncode == 0:
            js = json.loads(out.stdout or '{}')
            streams = js.get('streams') or []
            if streams and 'duration' in streams[0]:
                dur = float(streams[0]['duration'])
        if dur <= 0:
            # fallback
            dur = mp.VideoFileClip(path, audio=False).duration
        _DURATION_CACHE[path] = dur
        return dur
    except Exception:
        return 0.0

def _build_concat_reencode(visemes: List[str], silence_ranges: List[Tuple[int,int]], audio_duration: float, base_path: str, audio_path: str, final_output_path: str) -> bool:
    """Construct and run a single ffmpeg concat re-encode using original files.

    We approximate silence by repeating the default clip until covering each silence block.
    Speech visemes are appended in sequence. This avoids Python decoding & composition.
    """
    base_path = os.path.abspath(base_path)
    default_clip = os.path.join(base_path, "default.mp4")
    if not os.path.exists(default_clip):
        print("[avatar] No default.mp4 found; cannot use concat re-encode path")
        return False
    # Precompute durations
    default_dur = _probe_duration(default_clip) or 0.2
    if default_dur <= 0:
        print("[avatar] Invalid default clip duration")
        return False
    list_lines = []
    # Add silence coverage
    for start_ms, end_ms in silence_ranges:
        sil_dur = max(0.0, (end_ms - start_ms)/1000.0)
        if sil_dur <= 0: continue
        repeats = max(1, int((sil_dur / default_dur) + 0.999))
        for _ in range(repeats):
            list_lines.append(f"file '{default_clip}'")
    # Add speech visemes
    for v in visemes:
        path = os.path.join(base_path, f"{v}.mp4")
        if not os.path.exists(path):
            path = default_clip
        list_lines.append(f"file '{path}'")
    if not list_lines:
        list_lines.append(f"file '{default_clip}'")
    tmp_list = os.path.splitext(final_output_path)[0] + "_concat_re.txt"
    try:
        with open(tmp_list, 'w') as f:
            f.write("\n".join(list_lines))
        filter_args = []
        if SCALE_FILTER:
            filter_args = ["-vf", f"scale={SCALE_FILTER}"]
        cmd = [
            "ffmpeg","-y","-f","concat","-safe","0","-i", tmp_list,
            "-i", audio_path,
            *filter_args,
            "-c:v","libx264","-preset", DEFAULT_PRESET, "-crf", DEFAULT_CRF,
            "-pix_fmt","yuv420p","-c:a","aac","-shortest","-movflags","+faststart", final_output_path
        ]
        t0 = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[avatar] Concat re-encode path finished in {time.time()-t0:.2f}s (segments={len(list_lines)})")
        return True
    except Exception as e:
        print(f"[avatar] Concat re-encode path failed: {e}")
        return False


    except Exception as e:
        print(f"[avatar] Copy-mode failed: {e} -> falling back to re-encode path")
        return False


async def create_talking_head(text, audio_output_path, final_output_path, base_clip_path: Optional[str] = None, fast: bool | None = None):
    """
    Create a talking head video from text with audio.

    Args:
        text (str): The input text to convert to a talking head video.
        audio_output_path (str): The path to save the generated audio file.
        final_output_path (str): The path to save the final video file.
    """
    print("************************************************\ntext_to_viseme")
    visemes = text_to_visemes(text)
    print("viseme gen done, tts")
    await text_to_speech(text, audio_output_path)
    print("tts finished")

    audio = AudioSegment.from_file(audio_output_path)
    audio_duration = len(audio) / 1000
    print("detecting silence")
    silence_ranges = detect_silence_segments(audio_output_path)
    print("silence ranges detected, creating video from visemes ")

    # Allow caller to specify path; fallback to env or prior constant
    if base_clip_path is None:
        base_clip_path = os.getenv("AVATAR_VISEME_PATH", "/Users/ansh/Downloads/Work/HMS/repo/static/correct_crop/")
    # viseme_list = ['p', 't', 'S_cap', 'T_cap', 'f', 'k', 'i', 'l', 'r', 's', 'u', '@', 'a', 'o', 'e', 'O_cap', 'default']
    viseme_list = ['p', 't', 'S_cap', 'T_cap', 'f', 'k', 'i', 'l', 'r', 's', 'u', '@', 'a', 'o', 'e', 'O_cap', 'default']
    # Normalize source clips (if needed) to uniform codec/fps/size for safe copy-mode
    norm_base = _ensure_normalized_clips(base_clip_path, viseme_list)

    if not SYNC_MODE:
        # Attempt zero-reencode concat BEFORE loading clips (so slow preload not paid if copy works)
        if _can_use_copy_mode(visemes, norm_base, viseme_list):
            temp_base = os.path.splitext(final_output_path)[0] + "_tmp"
            if _build_copy_mode_video(visemes, norm_base, temp_base, audio_output_path, final_output_path):
                if _validate_media(audio_output_path, final_output_path, audio_duration):
                    return
                print("[avatar] Validation failed for concat output; removing and using Python assembly.")
                try:
                    os.remove(final_output_path)
                except Exception:
                    pass

        # Fast concat re-encode path (ffmpeg single pass) BEFORE Python assembly
        if CONCAT_REENCODE and not FORCE_REENCODE:
            if _build_concat_reencode(visemes, silence_ranges, audio_duration, norm_base, audio_output_path, final_output_path):
                if _validate_media(audio_output_path, final_output_path, audio_duration):
                    return
                else:
                    try: os.remove(final_output_path)
                    except Exception: pass
                    print("[avatar] Concat re-encode output failed validation; proceeding to Python assembly.")
    else:
        if DEBUG:
            print("[avatar] SYNC_MODE enabled: skipping copy & concat fast paths; using timing-aware assembly")

    # Lazy load only the visemes actually present (plus default) for Python path
    needed = sorted(set(visemes))
    if 'default' not in needed: needed.append('default')
    # Always pull clips from normalized base if it differs (improves decoding reliability & sync)
    load_path = norm_base if os.path.abspath(norm_base) != os.path.abspath(base_clip_path) else base_clip_path
    print(f"lazy loading {len(needed)} viseme clips (subset) from {load_path}{' (normalized)' if load_path==norm_base else ''}")
    viseme_clips = _lazy_load_required_visemes(needed, load_path)

    if SYNC_MODE:
        video = _build_timed_video(visemes, silence_ranges, audio_duration, viseme_clips)
    else:
        video = create_video_from_visemes(visemes, silence_ranges, audio_duration, viseme_clips)
    if video is None: raise RuntimeError("Failed to assemble video from visemes.")
    print("video from visemes finished (re-encode path)")
    video = video.without_audio()
    audio = mp.AudioFileClip(audio_output_path)
    print("syncing audio")
    final_video = video.set_audio(audio)
    print("audio synced, writing videofile (encode)")
    ffmpeg_params = ["-crf", DEFAULT_CRF]
    if DEFAULT_BITRATE:
        ffmpeg_params.extend(["-b:v", DEFAULT_BITRATE])
    t0 = time.time()
    final_video.write_videofile(
        final_output_path,
        codec=DEFAULT_CODEC,
        audio_codec="aac",
        threads=os.cpu_count() or 8,
        fps=DEFAULT_FPS,
        preset=DEFAULT_PRESET,
        ffmpeg_params=ffmpeg_params,
        logger=None,
        verbose=False,
    )
    print(f"[avatar] Re-encode path completed in {time.time()-t0:.2f}s")

    print("video file written\n********************************")

# Example usage
# phrase = ("Black spots on the skin could be due to a variety of reasons, ranging from harmless freckles"
#           "to more serious conditions like melanoma. It's important to get a comprehensive understanding"
#           "of your condition. Could you provide more information on the size, number, and location of the"
#           "black spots? Also, have you noticed any changes in the spots over time? Lastly, do you have any"
#           "genetic predispositions to skin conditions, for example, a variant in the MC1R gene (rs 1805007)"
#           "which is associated with an increased risk for melanoma?")

# phrase = "Black spots on the skin can arise from various causes, including hyperpigmentation, moles, or other dermatological conditions. It’s important to monitor any changes in size, shape, or color. Please consider seeing a dermatologist for a thorough evaluation, especially if you notice any rapid changes or if they are accompanied by other symptoms. In the meantime, can you tell me more about when you first noticed these spots and if you've experienced any other skin changes?"

# audio_output_path = "tts_audio.mp3"
# final_output_path = "talking_head.mp4"

# asyncio.run(create_talking_head(phrase, audio_output_path, final_output_path))