import os
from werkzeug.utils import secure_filename

ALLOWED_GENOMIC = {'.txt', '.tsv'}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_save(file_storage, base_dir: str, subdir: str) -> str:
    ensure_dir(os.path.join(base_dir, subdir))
    fname = secure_filename(file_storage.filename)
    ext = os.path.splitext(fname)[1].lower()
    full_dir = os.path.join(base_dir, subdir)
    full_path = os.path.join(full_dir, fname)
    file_storage.save(full_path)
    return full_path