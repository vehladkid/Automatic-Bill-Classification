"""Single entry point for local-LLM calls. Wraps a client (OllamaClient or a test double) with:
 * content-addressed result cache (same OCR/prompt/model/image => no second inference; HRBC_LLM_CACHE=0 disables; cache holds the model's JSON output only, never OCR text)
 * one inference at a time (bounded semaphore; protects 6 GB VRAM / RAM) with a wait timeout
 * bounded retry ONLY for transport failures/timeouts (a deterministic bad-JSON answer is not retried: temperature 0 would repeat it)
 * response schema validation (structure, not truth: values are still validated against the OCR text downstream)
 * logging of timing, token counts and failure class - never document text or extracted values."""
import json, time, hashlib, logging, threading, os
from pathlib import Path
from .client import LLMUnavailable, LLMBadOutput
log = logging.getLogger("llm")
PROMPT_VERSION = "p3"                 # bump when SYSTEM/USER_T change so stale cache entries are never reused
_SEM = threading.BoundedSemaphore(1)

def validate_schema(js):
    """Structural check of the model answer. Raises LLMBadOutput. Allowed: fields -> {name: {value, evidence} | scalar | null}; ocr_corrections -> list of dicts; classification -> dict; expense_details -> dict."""
    if not isinstance(js, dict): raise LLMBadOutput("answer is not a JSON object")
    f = js.get("fields")
    if f is not None and not isinstance(f, dict): raise LLMBadOutput("'fields' must be an object")
    for k, t in (("ocr_corrections", list), ("classification", dict), ("expense_details", dict)):
        if js.get(k) is not None and not isinstance(js[k], t): js[k] = None            # tolerate a malformed optional section: drop it rather than fail the whole answer
    if "fields" not in js and "classification" not in js: raise LLMBadOutput("answer has neither 'fields' nor 'classification'")   # {"fields": {}} is valid: it means nothing was found
    return js

class LLMService:
    def __init__(self, client, cache_dir=None, use_cache=None, retries=1, wait_timeout=300):
        self.client, self.retries, self.wait_timeout = client, retries, wait_timeout
        self.s = client.s; self.use_cache = (os.environ.get("HRBC_LLM_CACHE", "1") == "1") if use_cache is None else use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(os.environ.get("HRBC_OUT", Path(__file__).resolve().parents[2] / "outputs")) / "llm_cache"
        self.last_stats = {}; self.last_cached = False
    def _key(self, system, user, image):
        h = hashlib.sha256(); [h.update(str(x).encode()) for x in (PROMPT_VERSION, self.s.get("model"), self.s.get("use_image"), self.s.get("max_tokens"), system, user)]
        if image is not None and self.s.get("use_image"): h.update(image.convert("L").resize((64, 64)).tobytes())
        return h.hexdigest()
    def chat_json(self, system, user, image=None):
        """-> (parsed_json, raw_text, seconds). Raises LLMUnavailable / LLMBadOutput."""
        key = self._key(system, user, image); f = self.cache_dir / f"{key}.json"
        if self.use_cache and f.exists():
            try: d = json.loads(f.read_text()); self.last_cached = True; self.last_stats = d.get("stats", {}); log.info("llm cache hit"); return d["js"], "", 0.0
            except Exception: pass
        self.last_cached = False
        if not _SEM.acquire(timeout=self.wait_timeout): raise LLMUnavailable("LLM busy: timed out waiting for the single inference slot")
        try:
            last = None
            for attempt in range(self.retries + 1):
                try:
                    js, raw, secs = self.client.chat_json(system, user, image); js = validate_schema(js)
                    self.last_stats = dict(getattr(self.client, "last_stats", {}) or {})
                    log.info("llm ok %.1fs prompt_tokens=%s output_tokens=%s attempt=%d", secs, self.last_stats.get("prompt_eval_count"), self.last_stats.get("eval_count"), attempt)
                    if self.use_cache:
                        self.cache_dir.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(js=js, stats=self.last_stats))); tmp.replace(f)
                    return js, raw, secs
                except LLMUnavailable as e:
                    last = e; log.warning("llm transport failure (%s) attempt=%d", type(e).__name__, attempt); time.sleep(min(2 ** attempt, 5))
                except LLMBadOutput as e:
                    log.warning("llm bad output: %s", e); raise
            raise last
        finally: _SEM.release()
