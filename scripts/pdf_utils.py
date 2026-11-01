"""PDF -> page-1 JPEG (110 dpi) + word boxes from the PDF text layer (pdftotext -bbox). Poppler only; cached."""
import re, subprocess, html
from pathlib import Path
DPI = 110

def rasterize(pdf, out_jpg):
    out_jpg = Path(out_jpg)
    if not out_jpg.exists():
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "1", "-l", "1", "-jpeg", "-jpegopt", "quality=90", "-singlefile", str(pdf), str(out_jpg.with_suffix(""))], check=True, capture_output=True, timeout=120)
    return out_jpg

def words(pdf):
    """Words of page 1 with pixel boxes at DPI. Empty list if the PDF has no text layer."""
    r = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "-bbox", str(pdf), "-"], capture_output=True, text=True, timeout=120).stdout
    s = DPI / 72
    return [{"text": html.unescape(m[4]), "bbox": [int(float(m[0]) * s), int(float(m[1]) * s), int(float(m[2]) * s), int(float(m[3]) * s)]}
            for m in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', r)]

def pages(pdf):
    r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", r); return int(m.group(1)) if m else None
