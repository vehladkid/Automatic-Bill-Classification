"""Test infrastructure: a minimal IMAP4rev1 server (TLS) plus a throw-away PKI built with the openssl CLI. Local only (127.0.0.1). Used by the ingestion hardening tests and the audit smoke test.
Supports LOGIN, AUTHENTICATE XOAUTH2, SELECT/EXAMINE (+UIDVALIDITY), UID SEARCH (ALL | UID a:b | UID a:* with the real '*' quirk | HEADER Message-ID), UID FETCH (header / full body / RFC822.SIZE), UID STORE, LOGOUT.
Failure injection: login_error, fetch_fail_uids, drop_on_fetch_uids. `log` records every command verb so tests can prove the client never sends STORE/EXPUNGE/COPY/DELETE by default."""
import re, ssl, socket, threading, subprocess, base64, shlex, tempfile
from pathlib import Path

def make_pki(d=None):
    d = Path(d or tempfile.mkdtemp(prefix="pki_")); d.mkdir(parents=True, exist_ok=True); run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, check=True)
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem", "-days", "2", "-subj", "/CN=Test CA")
    (d / "ca.cnf").write_text("[ca]\ndefault_ca=CA_default\n[CA_default]\ndir=.\ndatabase=index.txt\nnew_certs_dir=.\nserial=serial\ndefault_md=sha256\npolicy=pol\ncopy_extensions=copy\nunique_subject=no\n[pol]\ncommonName=supplied\n"); (d / "index.txt").write_text(""); (d / "serial").write_text("01\n")
    def issue(name, san, cn="localhost", dates=None):
        run("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", f"{name}.key", "-out", f"{name}.csr", "-subj", f"/CN={cn}", "-addext", f"subjectAltName={san}")
        extra = ["-startdate", dates[0], "-enddate", dates[1]] if dates else ["-days", "1"]
        run("openssl", "ca", "-batch", "-config", "ca.cnf", "-cert", "ca.pem", "-keyfile", "ca.key", "-in", f"{name}.csr", "-out", f"{name}.pem", "-notext", *extra)
        return str(d / f"{name}.pem"), str(d / f"{name}.key")
    pki = dict(ca=str(d / "ca.pem"), good=issue("good", "DNS:localhost,IP:127.0.0.1"), mismatch=issue("mismatch", "DNS:other.example", cn="other.example"), expired=issue("expired", "DNS:localhost", dates=("20200101000000Z", "20200102000000Z")))
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "self.key", "-out", "self.pem", "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1")
    pki["selfsigned"] = (str(d / "self.pem"), str(d / "self.key")); return pki

class StubServer:
    def __init__(self, cert, messages=None, uidvalidity=7, user="user", password="pw", tokens=()):
        self.cert = cert; self.mbox = []; self.uidvalidity = uidvalidity; self.user, self.password, self.tokens = user, password, set(tokens)
        self.log = []; self.login_error = None; self.fetch_fail_uids = set(); self.drop_on_fetch_uids = set(); self.stored = []; self.connections = 0
        for m in messages or []: self.add(m)
    def add(self, m): self.mbox.append(m if isinstance(m, bytes) else m.as_bytes()); return len(self.mbox)
    def start(self):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(*self.cert); self.srv = socket.socket(); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); self.srv.bind(("127.0.0.1", 0)); self.srv.listen(8); self.port = self.srv.getsockname()[1]; self._ctx = ctx; self._stop = False
        threading.Thread(target=self._accept, daemon=True).start(); return self.port
    def stop(self):
        self._stop = True
        try: self.srv.close()
        except Exception: pass
    def _accept(self):
        while not self._stop:
            try: c, _ = self.srv.accept()
            except OSError: return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()
    def _serve(self, raw):
        try: s = self._ctx.wrap_socket(raw, server_side=True)
        except Exception: raw.close(); return                       # client rejected our certificate
        self.connections += 1; f = s.makefile("rwb", buffering=0); w = lambda b: f.write(b if isinstance(b, bytes) else b.encode()); w("* OK ready\r\n"); readonly = False
        try:
            while True:
                line = f.readline()
                if not line: return
                tag, _, rest = line.decode().strip().partition(" "); verb = rest.split(" ")[0].upper()
                self.log.append(verb + ((" " + rest.split(" ")[1].upper()) if verb == "UID" else ""))
                if verb == "CAPABILITY": w("* CAPABILITY IMAP4rev1 AUTH=XOAUTH2\r\n"); w(f"{tag} OK done\r\n")
                elif verb == "LOGIN":
                    args = shlex.split(rest)[1:]
                    if self.login_error == "throttle": w(f"{tag} NO [THROTTLED] Too many login attempts\r\n")
                    elif self.login_error == "authfail" or args != [self.user, self.password]: w(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials\r\n")
                    else: w(f"{tag} OK logged in\r\n")
                elif verb == "AUTHENTICATE":
                    w("+ \r\n"); blob = base64.b64decode(f.readline().strip() or b""); ok = False
                    m = re.match(rb"user=(.*?)\x01auth=Bearer (.*?)\x01\x01$", blob)
                    if m and m.group(1).decode() == self.user and m.group(2).decode() in self.tokens and not self.login_error: ok = True
                    if ok: w(f"{tag} OK authenticated\r\n")
                    else: w("+ eyJzdGF0dXMiOiI0MDEifQ==\r\n"); f.readline(); w(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials\r\n")
                elif verb in ("SELECT", "EXAMINE"):
                    readonly = verb == "EXAMINE"; w(f"* {len(self.mbox)} EXISTS\r\n"); w(f"* OK [UIDVALIDITY {self.uidvalidity}] UIDs valid\r\n"); w(f"{tag} OK [{'READ-ONLY' if readonly else 'READ-WRITE'}] done\r\n")
                elif verb == "UID":
                    parts = rest.split(" "); sub = parts[1].upper(); top = len(self.mbox)
                    if sub == "SEARCH":
                        crit = " ".join(parts[2:]); uids = list(range(1, top + 1))
                        m = re.match(r"UID (\d+):(\d+|\*)$", crit)
                        if m:
                            lo = int(m.group(1)); hi = top if m.group(2) == "*" else int(m.group(2)); lo = min(lo, top) if m.group(2) == "*" else lo      # IMAP quirk: n:* always includes the highest UID
                            uids = [u for u in uids if lo <= u <= hi]
                        mh = re.match(r'HEADER Message-ID "(.*)"$', crit, re.I)
                        if mh: uids = [u for u in uids if mh.group(1).encode() in self.mbox[u - 1].split(b"\n\n")[0].split(b"\r\n\r\n")[0]]
                        w("* SEARCH " + " ".join(map(str, uids)) + "\r\n"); w(f"{tag} OK done\r\n")
                    elif sub == "FETCH":
                        uid = int(parts[2]); raw_msg = self.mbox[uid - 1] if 0 < uid <= top else None
                        if uid in self.drop_on_fetch_uids: s.close(); return
                        if raw_msg is None or uid in self.fetch_fail_uids: w(f"{tag} NO [SERVERBUG] fetch failed\r\n"); continue
                        what = rest.upper()
                        if "HEADER" in what: data = re.split(rb"\r?\n\r?\n", raw_msg, 1)[0] + b"\r\n\r\n"; w(f"* {uid} FETCH (UID {uid} RFC822.SIZE {len(raw_msg)} BODY[HEADER] {{{len(data)}}}\r\n")
                        else: data = raw_msg; w(f"* {uid} FETCH (UID {uid} BODY[] {{{len(data)}}}\r\n")
                        w(data); w(")\r\n"); w(f"{tag} OK done\r\n")
                    elif sub == "STORE":
                        if readonly: w(f"{tag} NO [READ-ONLY] mailbox is read-only\r\n")
                        else: self.stored.append(parts[2]); w(f"{tag} OK done\r\n")
                elif verb == "LOGOUT": w("* BYE\r\n"); w(f"{tag} OK bye\r\n"); s.close(); return
                else: w(f"{tag} OK\r\n")
        except Exception: return
