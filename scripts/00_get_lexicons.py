"""Download the word lists into data/lexicons/.

    python scripts/00_get_lexicons.py                  # what the assignment needs
    python scripts/00_get_lexicons.py --with-harvard   # plus the extra-credit list

Sources
  Loughran-McDonald Master Dictionary  https://sraf.nd.edu/loughranmcdonald-master-dictionary/
  Harvard General Inquirer (optional)  https://inquirer.sites.fas.harvard.edu/

Free. If a link has rotted since this was written, go to the page above, download
by hand, and drop the file in data/lexicons/ under the name in src/config.py.
Do not silently substitute a different word list.
"""

import sys
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    HARVARD_GI_PATH, HARVARD_GI_URL, LM_MASTER_DICT_PATH, LM_MASTER_DICT_URL,
)

# Exact downloaded source bytes used for this submission; upstream drift is explicit.
LM_EXPECTED_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"

def validate_lm(path, url, retrieved=False):
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != LM_EXPECTED_SHA256:
        raise ValueError(f"Dictionary bytes changed ({digest}); restore the documented version before comparing results")
    from src.lexicons import load_master_dictionary, lm_word_lists
    master = load_master_dictionary(path)
    info = {"source_url": url, "sha256": digest, "rows": len(master),
            "active_sizes": {k: len(v) for k,v in lm_word_lists(master).items()},
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    sidecar = path.with_suffix(".json")
    if retrieved:
        info["retrieved_at_utc"] = info["checked_at_utc"]
    elif sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous.get("sha256") == digest and "retrieved_at_utc" in previous:
            info["retrieved_at_utc"] = previous["retrieved_at_utc"]
    sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")

REQUIRED = [
    ("Loughran-McDonald Master Dictionary", LM_MASTER_DICT_URL, LM_MASTER_DICT_PATH),
]
OPTIONAL = [
    ("Harvard General Inquirer (inqtabs.txt)", HARVARD_GI_URL, HARVARD_GI_PATH),
]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-harvard", action="store_true",
                    help="also fetch the Harvard General Inquirer (extra credit only)")
    args = ap.parse_args()
    targets = REQUIRED + (OPTIONAL if args.with_harvard else [])

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (course assignment)"})
    failures = 0
    for name, url, path in targets:
        if path.exists() and path.stat().st_size > 10_000:
            if path == LM_MASTER_DICT_PATH:
                validate_lm(path, url)
            print(f"[skip] {name} already at {path} ({path.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"[get ] {name}")
        try:
            resp = session.get(url, timeout=180)
            resp.raise_for_status()
            if path == LM_MASTER_DICT_PATH and hashlib.sha256(resp.content).hexdigest() != LM_EXPECTED_SHA256:
                raise ValueError("Downloaded dictionary differs from the documented source checksum; no replacement saved")
            path.write_bytes(resp.content)
            if path == LM_MASTER_DICT_PATH:
                validate_lm(path, url, retrieved=True)
            print(f"[ok  ] {path} ({len(resp.content)/1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name}: {exc}\n       Download it by hand and save it to {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
