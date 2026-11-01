"""TLS for mail connections: certificate AND hostname verification are always on. Extra trust (a private CA for a local test server) is accepted only outside prod and only via an explicit CA file; there is no option to disable verification."""
import ssl
from . import config
class TlsConfigError(Exception): pass
def context(ca_file=None):
    ctx = ssl.create_default_context()                    # CERT_REQUIRED + check_hostname + system trust store
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if ca_file:
        if config.env_name() == "prod": raise TlsConfigError("extra TLS trust is not allowed in prod")
        ctx.load_verify_locations(cafile=ca_file)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx
