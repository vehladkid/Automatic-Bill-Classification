"""LLM settings from environment variables. Default OFF. Employee documents are sent only to a loopback address unless the operator sets BOTH
HRBC_LLM_ALLOW_REMOTE=1 and HRBC_LLM_REMOTE_ACK=<the exact acknowledgement string below>."""
import os
from urllib.parse import urlparse
ACK = "I_ACCEPT_THAT_EMPLOYEE_DOCUMENTS_LEAVE_THIS_MACHINE"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

class PrivacyError(Exception): pass

def settings(env=None):
    e = os.environ if env is None else env
    return dict(mode=e.get("HRBC_LLM_MODE", "off").lower(),                    # off | auto (only when needed) | always
                url=e.get("HRBC_LLM_URL", "http://127.0.0.1:11434"), model=e.get("HRBC_LLM_MODEL", "qwen2.5vl:3b"),
                use_image=e.get("HRBC_LLM_USE_IMAGE", "0") == "1", timeout=float(e.get("HRBC_LLM_TIMEOUT", "180")),
                allow_remote=e.get("HRBC_LLM_ALLOW_REMOTE", "0") == "1", remote_ack=e.get("HRBC_LLM_REMOTE_ACK", ""),
                max_tokens=int(e.get("HRBC_LLM_MAX_TOKENS", "1024")), keep_alive=e.get("HRBC_LLM_KEEP_ALIVE", "0"), llm_only=e.get("HRBC_LLM_ONLY_POLICY", "suggest"), max_ocr_chars=int(e.get("HRBC_LLM_MAX_OCR_CHARS", "6000")), image_long_side=int(e.get("HRBC_LLM_IMAGE_SIDE", "1280")))

def check_privacy(s):
    """Raises PrivacyError unless the endpoint is local or remote use was explicitly acknowledged."""
    host = (urlparse(s["url"]).hostname or "").lower()
    if host in LOOPBACK: return "local"
    if s["allow_remote"] and s["remote_ack"] == ACK: return "remote-acknowledged"
    raise PrivacyError(f"LLM endpoint host '{host}' is not local. Employee documents are not sent off this machine unless HRBC_LLM_ALLOW_REMOTE=1 and HRBC_LLM_REMOTE_ACK={ACK} are both set.")
