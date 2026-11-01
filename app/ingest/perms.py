"""Least-privilege permissions for everything the application persists under HRBC_OUT (default <repo>/outputs): ingest.db (+WAL/SHM), hr.db, raw inbox copies, per-document results (record.json, OCR text, page images), classified/unclassified originals, LLM cache, worker lock.
Rule: directories 0700, files 0600, nothing readable by group/other. `audit()` lists violations, `fix()` repairs them, `apply_umask()` makes new files private."""
import os, stat
from pathlib import Path
def apply_umask(): os.umask(0o077)
def private_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(p, 0o700); return p
def private_file(p):
    p = Path(p)
    if p.exists(): os.chmod(p, 0o600)
def secure_db(path):
    """db file plus its -wal/-shm sidecars"""
    for suf in ("", "-wal", "-shm"): private_file(str(path) + suf)
def audit(root):
    bad = []; root = Path(root)
    if not root.exists(): return bad
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = Path(dp) / n
            try: m = stat.S_IMODE(p.lstat().st_mode)
            except OSError: continue
            if stat.S_ISLNK(p.lstat().st_mode): continue
            if m & 0o077: bad.append((str(p.relative_to(root)), oct(m)))
    try:
        m = stat.S_IMODE(root.stat().st_mode)
        if m & 0o077: bad.insert(0, (".", oct(m)))
    except OSError: pass
    return bad
def fix(root):
    n = 0; root = Path(root)
    for dp, dns, fns in os.walk(root):
        for name in dns:
            p = Path(dp) / name
            if not p.is_symlink(): os.chmod(p, 0o700); n += 1
        for name in fns:
            p = Path(dp) / name
            if not p.is_symlink(): os.chmod(p, 0o600); n += 1
    if root.exists(): os.chmod(root, 0o700)
    return n
