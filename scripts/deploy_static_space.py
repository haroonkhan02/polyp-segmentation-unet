"""Publish the in-browser demo: weights and ONNX files to a model repo, web/ to a static Space.

Uses the Hugging Face login stored by `hf auth login`. The page loads the INT8 ONNX files from
the model repo, so the Space itself only holds HTML, JavaScript and five example test frames
(with their expert masks as PNG).

Example:
    python scripts/deploy_static_space.py --dry-run   # build outputs/static_space only
    python scripts/deploy_static_space.py --models    # also upload weights and ONNX files
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from polypseg.config import KVASIR_DIR, MASK_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "haroonx02/polyp-segmentation-unet-resnet34"
SPACE_REPO = "haroonx02/polyp-segmentation-browser"
WEB_FILES = ("index.html", "app.js", "polyp.js", "README.md")
# Test-split frames: typical, small, large, and two of the four worst cases.
EXAMPLES = {"1_typical": "cju5wkonqlrl409877y8zvnub", "2_small": "cju7dvl5m2n4t0755hlnnjjet",
            "3_large": "cju7eotqi2qea0871y8yc7tqh", "4_partial_miss": "cju2nbdpmlmcj0993s1cht0dz",
            "5_confident_miss": "cju2zy1e49pqk0878t6ncqn12"}
HUB_FILES = {"onnx/polyp_unet_int8.onnx": "models/onnx/polyp_unet_int8.onnx",
             "onnx/polyp_unet_fp16.onnx": "models/onnx/polyp_unet_fp16.onnx",
             "onnx/polyp_unet_mc8_int8.onnx": "models/onnx/polyp_unet_mc8_int8.onnx",
             "unet_resnet34.pt": "models/unet_resnet34.pt",
             "README.md": "hub/MODEL_CARD_HUB.md"}


def build_bundle(out: Path, root: Path = KVASIR_DIR) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "examples").mkdir(parents=True)
    for name in WEB_FILES:
        shutil.copy(ROOT / "web" / name, out / name)
    for name, stem in EXAMPLES.items():
        shutil.copy(root / "images" / f"{stem}.jpg", out / "examples" / f"{name}.jpg")
        mask = cv2.imread(str(root / "masks" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(out / "examples" / f"{name}_mask.png"),
                    ((mask > MASK_THRESHOLD) * 255).astype("uint8"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=SPACE_REPO)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "static_space")
    parser.add_argument("--models", action="store_true", help="upload weights + ONNX files too")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(args.out)
    print(f"Bundle ready at {bundle}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    if args.models:
        api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        for dest, src in HUB_FILES.items():
            api.upload_file(path_or_fileobj=ROOT / src, path_in_repo=dest, repo_id=MODEL_REPO,
                            commit_message=f"Add {Path(dest).name}")
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=bundle,
                      commit_message="Deploy browser demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
