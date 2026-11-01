import sys, os, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app", ROOT / "scripts", ROOT / "scripts" / "taskB"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
os.environ.setdefault("HRBC_OUT", tempfile.mkdtemp(prefix="hrbc_test_"))   # never touch the real outputs/
