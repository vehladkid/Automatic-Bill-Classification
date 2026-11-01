"""Minimal Ollama-compatible chat client (stdlib + requests). Temperature 0, JSON output. Raises LLMUnavailable on any transport problem."""
import base64, io, json, re, time
from .config import settings, check_privacy

class LLMUnavailable(Exception): pass
class LLMBadOutput(Exception): pass

def parse_json(txt):
    t = str(txt or "").strip(); t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    sv = salvage_truncated(t)
    if sv is not None: return sv
    raise LLMBadOutput("model did not return valid JSON")

def salvage_truncated(t):
    """Output cut off mid-way (e.g. a repetition loop hit the token cap): cut back to the last position where closing the open brackets gives valid JSON.
    Only complete earlier fields survive; the damaged tail is dropped. Values are still validated against the OCR text downstream. Returns dict or None."""
    for cut in range(len(t), max(0, len(t) - 6000), -1):
        if t[cut - 1:cut] not in (",", "\n", "}", "]"): continue
        head = t[:cut].rstrip().rstrip(",")
        stack, instr, esc = [], False, False
        for ch in head:
            if instr:
                esc = (not esc) and ch == "\\"
                if ch == '"' and not esc: instr = False
                continue
            if ch == '"': instr = True
            elif ch in "{[": stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack: stack.pop()
        if instr: continue
        try:
            v = json.loads(head + "".join(reversed(stack)))
            if isinstance(v, dict) and v: return v
        except Exception: continue
    return None

class OllamaClient:
    def __init__(self, s=None):
        self.s = s or settings(); self.where = check_privacy(self.s); self.last_stats = {}

    def available(self):
        import requests
        try:
            r = requests.get(self.s["url"].rstrip("/") + "/api/tags", timeout=3); r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return (self.s["model"] in names or any(n.split(":")[0] == self.s["model"].split(":")[0] and ":" not in self.s["model"] for n in names)), names
        except Exception as e: return False, [f"{type(e).__name__}: {e}"]

    def chat_json(self, system, user, image=None):
        """image: PIL.Image or None. Returns (parsed_json, raw_text, seconds)."""
        import requests
        msg = {"role": "user", "content": user}
        if image is not None and self.s["use_image"]:
            im = image.convert("RGB"); sc = self.s["image_long_side"] / max(im.size)
            if sc < 1: im = im.resize((round(im.width * sc), round(im.height * sc)))
            b = io.BytesIO(); im.save(b, "JPEG", quality=90); msg["images"] = [base64.b64encode(b.getvalue()).decode()]
        body = {"model": self.s["model"], "messages": [{"role": "system", "content": system}, msg], "stream": False, "format": "json", "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": self.s.get("max_tokens", 1024)}, "keep_alive": self.s.get("keep_alive", "0")}
        t0 = time.time()
        try:
            r = requests.post(self.s["url"].rstrip("/") + "/api/chat", json=body, timeout=self.s["timeout"]); r.raise_for_status()
            j = r.json(); raw = j["message"]["content"]
            self.last_stats = {k: j.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")}
        except Exception as e: raise LLMUnavailable(f"{type(e).__name__}: {e}")
        return parse_json(raw), raw, round(time.time() - t0, 2)
