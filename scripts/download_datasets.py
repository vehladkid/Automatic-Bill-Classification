"""Download public datasets into datasets/public/. Idempotent. `--cord-train` adds the 1.8GB CORD train shards."""
import sys, subprocess, zipfile, urllib.request
from common import PUBLIC, ROOT

def kaggle(slug, out):
    if not any(f.is_file() for f in out.glob("**/*")):
        subprocess.run([str(ROOT / ".venv/bin/kaggle"), "datasets", "download", "-d", slug, "-p", str(out), "--unzip"], check=True)

def hf(repo, out, patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=patterns)

if __name__ == "__main__":
    # SROIE 2019 (ICDAR) was fetched manually earlier: datasets/public/sroie/sroie-datasetv2.zip (+ extracted/)
    hf("nielsr/rvl_cdip_10_examples_per_class", PUBLIC / "rvl_cdip_160", ["data/*"])  # 16 classes x 10 imgs; the full set is ~100GB, non-expense negatives only need a sample
    z = PUBLIC / "funsd" / "dataset.zip"  # official source (Jaume et al. 2019)
    if not z.exists():
        z.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://guillaumejaume.github.io/FUNSD/dataset.zip", z)
        zipfile.ZipFile(z).extractall(z.parent)
    hf("naver-clova-ix/cord-v2", PUBLIC / "cord", ["data/validation*", "data/test*"] + (["data/train*"] if "--cord-train" in sys.argv else []))
    kaggle("surajitsadhukhan/indian-grocery-tax-invoice-image-dataset", PUBLIC / "indian_grocery_gst")
    kaggle("mehaksingal/personal-financial-dataset-for-india", PUBLIC / "in_financial_docs")  # Indian bank/salary/ITR/cheque (negatives) + mixed-country utility bills
    hf("destryet/medical_bills", PUBLIC / "hf_medical_bills", ["*.pdf"])  # license not stated -> flagged in manifest
    kaggle("jenswalter/receipts", PUBLIC / "my_receipts")  # CC0, 1252 real receipts (PDF) in country/category folders
    # ReceiptBench subset + InvoicesReceiptsPT subset: run acquire_real_foreign.py (diverse subsets only, remote-zip range reads)
