"""Attachment safety checks. Allow-list by FILE SIGNATURE (never trust the extension or the declared MIME type alone). Archives/office/script files are rejected outright, which
also removes the archive-bomb surface. Every failure has a stable code + an actionable message."""
import re, subprocess, io, os, tempfile
from pathlib import Path
MAX_BYTES = int(os.environ.get("HRBC_INGEST_MAX_BYTES", 20 * 1024 * 1024)); MAX_PAGES = int(os.environ.get("HRBC_INGEST_MAX_PAGES", 10))
MAX_ATTACHMENTS = int(os.environ.get("HRBC_INGEST_MAX_ATTACHMENTS", 10)); MAX_TOTAL = int(os.environ.get("HRBC_INGEST_MAX_TOTAL_BYTES", 50 * 1024 * 1024))
MAX_PIXELS = 60_000_000; MIN_BYTES = 1500          # tiny inline images (signature logos, tracking pixels) are ignored, not failed
SIG = [(b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"%PDF-", "pdf")]
DENY_SIG = [(b"PK\x03\x04", "zip/office archive"), (b"Rar!", "rar archive"), (b"\x1f\x8b", "gzip"), (b"7z\xbc\xaf", "7z archive"), (b"MZ", "executable"), (b"\x7fELF", "executable"), (b"#!", "script"), (b"<html", "html"), (b"<!DOC", "html")]
EXT = {"jpeg": {".jpg", ".jpeg"}, "png": {".png"}, "pdf": {".pdf"}}; MIME = {"jpeg": {"image/jpeg", "image/jpg", "image/pjpeg"}, "png": {"image/png"}, "pdf": {"application/pdf"}}
class ValidationUnavailable(Exception):
    """an environment problem (e.g. pdfinfo missing), NOT a property of the attachment: never marks the file FAILED, the message is retried"""
class AttachmentError(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code; self.msg = msg

def sniff(data):
    for s, k in SIG:
        if data[:len(s)] == s: return k
    return None

def validate(data, filename, mime):
    """-> (kind, pages). Raises AttachmentError(code, message)."""
    if not data: raise AttachmentError("empty", "The attachment is empty.")
    if len(data) > MAX_BYTES: raise AttachmentError("too_large", f"Attachment is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES // 1_000_000} MB. Send a smaller scan.")
    kind = sniff(data)
    if kind is None:
        for s, why in DENY_SIG:
            if data[:len(s)].lower() == s.lower(): raise AttachmentError("unsupported_type", f"{why} files are not accepted. Send the bill as PDF, JPG or PNG.")
        raise AttachmentError("unsupported_type", "Unrecognised file content. Only PDF, JPG and PNG are accepted.")
    ext = Path(str(filename or "")).suffix.lower()
    if ext and ext not in EXT[kind]: raise AttachmentError("type_mismatch", f"File name says '{ext}' but the content is {kind.upper()}; rename it correctly and resend.")
    m = (mime or "").lower().split(";")[0].strip()
    if m and m not in MIME[kind] | {"application/octet-stream"}: raise AttachmentError("type_mismatch", f"Declared type '{m}' does not match the {kind.upper()} content.")
    if kind == "pdf":
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.pdf"; p.write_bytes(data)
            try: r = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired: raise AttachmentError("corrupt", "The PDF took too long to inspect (possibly malformed).")
            except OSError as e: raise ValidationUnavailable(f"pdfinfo unavailable ({type(e).__name__})")
        txt = r.stdout + r.stderr
        if re.search(r"Encrypted:\s+yes", txt, re.I) or "Incorrect password" in txt: raise AttachmentError("encrypted", "The PDF is password-protected or encrypted. Remove the password and resend.")
        m = re.search(r"Pages:\s+(\d+)", r.stdout)
        if r.returncode != 0 or not m: raise AttachmentError("corrupt", "The PDF could not be read (corrupt or malformed).")
        n = int(m.group(1))
        if n > MAX_PAGES: raise AttachmentError("too_many_pages", f"PDF has {n} pages; the limit is {MAX_PAGES}. Send the bill pages only.")
        return kind, n
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as im: im.verify()
        with Image.open(io.BytesIO(data)) as im: w, h = im.size
    except Exception as e: raise AttachmentError("corrupt", f"The image could not be decoded ({type(e).__name__}); it may be corrupt or too large.")
    if w * h > MAX_PIXELS: raise AttachmentError("too_large", "Image resolution is too large.")
    return kind, 1
