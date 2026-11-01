"""Sender / recipient authorisation policy (fail closed).
AUTHORISATION = the sender address is on the explicit employee allow-list (table `senders`; one employee may have several rows = aliases) AND the message was addressed to the expense mailbox or a configured alias.
AUTHENTICATION (SPF/DKIM/DMARC from Authentication-Results) is supporting evidence only: it can only make an allow-listed claim acceptable, never create one, and it must come from OUR receiving server (authserv-id) and be ALIGNED with the From domain
(a pass for the attacker's own domain proves nothing about a claimed employee From). Only the OUTER message's headers are ever used: forwarded/embedded messages and body text are never trusted for identity."""
import re
from email.utils import parseaddr
from . import config, db
PUBLIC_SUFFIXES = {"co.uk", "com.au", "co.in", "org.in", "gov.in", "net.in", "ac.in", "co.nz", "com.br", "co.za", "com.sg", "co.jp"}
def dom(addr): return addr.rsplit("@", 1)[1].lower() if "@" in (addr or "") else ""
def _registrable(d):
    ps = d.split(".")
    if len(ps) < 2: return ""
    return ".".join(ps[-3:]) if ".".join(ps[-2:]) in PUBLIC_SUFFIXES and len(ps) >= 3 else ".".join(ps[-2:])
def aligned(a, b):
    """relaxed alignment: same organisational domain (approximation without the full public-suffix list; the small built-in list covers common multi-label suffixes)"""
    a, b = (a or "").lower().strip("."), (b or "").lower().strip(".")
    if not a or not b or "." not in a or "." not in b: return False
    if a in PUBLIC_SUFFIXES or b in PUBLIC_SUFFIXES: return False
    return a == b or _registrable(a) == _registrable(b) and _registrable(a) != ""
def parse_results(headers, authserv):
    """-> list of (method, result, domain, prop) from Authentication-Results headers written by `authserv` only"""
    out = []
    for h in headers or []:
        parts = [p.strip() for p in re.sub(r"\s+", " ", str(h)).split(";")]
        if not parts: continue
        sid = parts[0].split(" ")[0].lower()
        if not authserv or sid != authserv: continue
        for p in parts[1:]:
            m = re.match(r"(spf|dkim|dmarc)=(\w+)(.*)$", p, re.I)
            if not m: continue
            method, result, rest = m.group(1).lower(), m.group(2).lower(), m.group(3)
            props = dict((k.lower(), v.strip('"<>')) for k, v in re.findall(r"([\w.]+)=(\S+)", rest))
            d = props.get("header.from") if method == "dmarc" else props.get("header.d") if method == "dkim" else (props.get("smtp.mailfrom") or props.get("smtp.helo"))
            out.append((method, result, dom(d) if d and "@" in d else (d or "").lower(), props))
    return out
def evidence(headers, from_addr, authserv):
    """-> (kinds_of_aligned_pass, failed_dmarc). kinds are labels only (no addresses)"""
    fd = dom(from_addr); kinds = []; dmarc_fail = False
    for method, result, d, _ in parse_results(headers, authserv):
        if method == "dmarc" and result in ("fail", "reject", "quarantine"): dmarc_fail = True
        if result == "pass" and aligned(d, fd): kinds.append(f"{method}_aligned")
    return sorted(set(kinds)), dmarc_fail
def authorise(msg, extra_mailboxes=()):
    """-> (employee_id | None, reason). Reasons are stable codes. extra_mailboxes: the address of the operator-registered connected mailbox the message was read from (it is a legitimate recipient there); the sender allow-list still decides the employee."""
    pol = config.policy(); allowed_mailboxes = ({pol["mailbox"]} | set(pol["aliases"])) - {""}
    if not allowed_mailboxes: return None, "mailbox_not_configured"          # fail closed: the base configuration is still required
    allowed_mailboxes |= {str(x).strip().lower() for x in extra_mailboxes if x}
    if getattr(msg, "from_count", 1) != 1: return None, "ambiguous_from"
    rcpts = {r.lower() for r in (getattr(msg, "recipients", None) or [msg.recipient]) if r}
    if not (rcpts & allowed_mailboxes): return None, "not_addressed_to_expense_mailbox"
    sender = (msg.sender or "").lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", sender): return None, "invalid_sender"
    if pol["allowed_domains"] and dom(sender) not in pol["allowed_domains"]: return None, "sender_domain_not_allowed"
    r = db.conn().execute("SELECT employee_id FROM senders WHERE email=? AND active=1", (sender,)).fetchone()
    if not r: return None, "sender_not_authorised"
    if pol["require_auth"]:
        headers = getattr(msg, "auth_headers", None) or ([msg.auth_results] if msg.auth_results else [])
        kinds, dmarc_fail = evidence(headers, sender, pol["authserv"])
        if dmarc_fail: return None, "sender_authentication_failed"
        if not kinds: return None, "sender_not_authenticated"
    return r["employee_id"], "ok"
