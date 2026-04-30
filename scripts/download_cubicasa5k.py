"""Download the CubiCasa5K dataset.

Source: https://zenodo.org/record/2613548 (5GB).

This script is intentionally thin — the dataset is too large to vendor in the
repository. Run it on a machine that will be used for training.

Usage:
    python scripts/download_cubicasa5k.py --out data/cubicasa5k
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/record/2613548/files/cubicasa5k.zip"


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "cubicasa5k.zip"
    if not zip_path.exists():
        print(f"Downloading {ZENODO_URL} → {zip_path} (this is ~5GB) …")
        with urllib.request.urlopen(ZENODO_URL) as resp, open(zip_path, "wb") as f:
            buf = bytearray(1 << 20)
            view = memoryview(buf)
            total = 0
            while True:
                n = resp.readinto(view)
                if not n:
                    break
                f.write(view[:n])
                total += n
                if total % (50 << 20) < (1 << 20):
                    sys.stdout.write(f"\r  {total / (1 << 20):.0f} MiB")
                    sys.stdout.flush()
        print()
    else:
        print(f"Archive already downloaded: {zip_path}")

    extract_dir = out_dir / "extracted"
    if not extract_dir.exists():
        print(f"Extracting → {extract_dir} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print("Done.")
    else:
        print(f"Already extracted to {extract_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/cubicasa5k"), help="Output directory"
    )
    args = parser.parse_args()
    main(args.out)
