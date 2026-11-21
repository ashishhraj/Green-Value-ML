"""Download the three public datasets and print their SHA-256 checksums.

The synthetic retail dataset needs no download: it is regenerated from a
fixed seed by src/data.py (and cached to data/synthetic_retail.csv.gz).
"""
import hashlib
import sys
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import DATA_DIR, SOURCES  # noqa: E402

EXPECTED_ROWS = {"creditcard.csv": 284_807, "telco_churn.csv": 7_043, "german_credit.csv": 1_000}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        path = DATA_DIR / name
        if not path.exists():
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        status = "OK" if rows == EXPECTED_ROWS[name] else f"UNEXPECTED (expected {EXPECTED_ROWS[name]})"
        print(f"{name:20s} rows={rows:>7,d}  {status}  sha256={digest}")


if __name__ == "__main__":
    main()
