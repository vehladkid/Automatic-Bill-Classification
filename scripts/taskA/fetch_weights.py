"""Robust chunked download (4 MB Range requests, per-chunk retry) for flaky links. Usage: fetch_weights.py <hf_repo> <filename> <dest>"""
import sys, os, time, requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url
repo, fn, dest = sys.argv[1:4]; url = hf_hub_url(repo, fn); size = get_hf_file_metadata(url).size
os.makedirs(os.path.dirname(dest), exist_ok=True); have = os.path.getsize(dest) if os.path.exists(dest) else 0
if have > size: os.remove(dest); have = 0
print("size", size, "have", have, flush=True)
while have < size:
    end = min(have + (4 << 20), size) - 1
    try:
        r = requests.get(url, headers={"Range": f"bytes={have}-{end}"}, timeout=30, allow_redirects=True)
        if r.status_code in (200, 206) and len(r.content) == end - have + 1 or (r.status_code == 206 and len(r.content) > 0):
            with open(dest, "ab") as f: f.write(r.content[: end - have + 1])
            have += min(len(r.content), end - have + 1)
        else: raise RuntimeError(f"status {r.status_code} len {len(r.content)}")
    except Exception as e:
        print("retry", have, str(e)[:60], flush=True); time.sleep(2)
    if have % (20 << 20) < (4 << 20): print(f"{have/1e6:.0f}/{size/1e6:.0f} MB", flush=True)
print("DONE", os.path.getsize(dest))
