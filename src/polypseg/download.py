"""Download and extract Kvasir-SEG (CC BY 4.0, ~46 MB) from the Simula dataset server.

Example:
    polyp-download            # -> data/Kvasir-SEG/{images,masks}
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import DATA_DIR, KVASIR_DIR, KVASIR_NUM_IMAGES, KVASIR_URL


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` with a progress bar; skip if the file is complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https URL
        total = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"{dest.name} already downloaded")
            return dest
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name) as bar:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))
    return dest


def extract(zip_path: Path, target_dir: Path) -> Path:
    """Extract the archive; the zip contains a top-level ``Kvasir-SEG/`` folder."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return target_dir / "Kvasir-SEG"


def verify(root: Path) -> tuple[int, int]:
    images = sorted((root / "images").glob("*.jpg"))
    masks = {p.name for p in (root / "masks").glob("*.jpg")}
    missing = [p.name for p in images if p.name not in masks]
    if missing:
        raise SystemExit(f"{len(missing)} images have no mask, e.g. {missing[:3]}")
    return len(images), len(masks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=KVASIR_URL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if (KVASIR_DIR / "images").exists() and args.data_dir == DATA_DIR:
        n_img, _ = verify(KVASIR_DIR)
        print(f"Kvasir-SEG already present: {n_img} images in {KVASIR_DIR}")
        return
    zip_path = download(args.url, args.data_dir / "kvasir-seg.zip")
    root = extract(zip_path, args.data_dir)
    n_img, n_mask = verify(root)
    if n_img != KVASIR_NUM_IMAGES:
        print(f"warning: expected {KVASIR_NUM_IMAGES} images, found {n_img}", file=sys.stderr)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"Kvasir-SEG ready: {n_img} images, {n_mask} masks in {root}")


if __name__ == "__main__":
    main()
