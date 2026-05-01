"""
Download MVTec 3D-AD dataset categories.

Usage:
    python scripts/download_data.py --category bagel
    python scripts/download_data.py --all
    python scripts/download_data.py --category bagel --dest data/raw

NOTE: You must first register at https://www.mvtec.com/company/research/datasets/mvtec-3d-ad
      and replace DOWNLOAD_TOKEN below with the token from your download link.
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

# Base URL pattern — fill in your token after registering on the MVTec website
BASE_URL = "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_3d_anomaly_detection"

CATEGORIES = [
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
]


def download_category(category: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{category}.tar.xz"
    out_path = dest / f"{category}.tar.xz"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print(f"Downloading {category} → {out_path}")
    urllib.request.urlretrieve(url, out_path, reporthook=_progress)
    print(f"\nExtracting {out_path} ...")
    import tarfile
    with tarfile.open(out_path, "r:xz") as tar:
        tar.extractall(dest)
    print(f"Done: {dest / category}")


def _progress(block, block_size, total):
    downloaded = block * block_size
    pct = min(100, downloaded * 100 // total) if total > 0 else 0
    print(f"\r  {pct:3d}%  {downloaded // 1_000_000} MB / {total // 1_000_000} MB", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, help="Single category to download")
    parser.add_argument("--all", action="store_true", help="Download all categories")
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    args = parser.parse_args()

    dest = Path(args.dest)
    categories = CATEGORIES if args.all else ([args.category] if args.category else [])

    if not categories:
        parser.print_help()
        return

    for cat in categories:
        download_category(cat, dest)


if __name__ == "__main__":
    main()
