#!/usr/bin/env python3
import argparse
from pathlib import Path
from datasets import load_from_disk
from PIL import ImageOps

def main():
    p = argparse.ArgumentParser(
        description="Export images from a HuggingFace split to PNG with 300 dpi and auto‐contrast"
    )
    p.add_argument("--split_dir", type=str, required=True,
                   help="Path to the dataset split directory (e.g. data/processed/test)")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory where PNG images will be saved")
    args = p.parse_args()

    ds = load_from_disk(args.split_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for idx, ex in enumerate(ds):
        img = ex["image"]  # PIL Image
        img = img.convert("RGB")                      # ensure 3-ch
        img = ImageOps.autocontrast(img, cutoff=2)    # boost contrast
        # save with DPI metadata so tesseract uses 300 dpi
        img.save(out_path / f"{idx:05d}.png", dpi=(300, 300))

    print(f"Exported {len(ds)} images to {out_path}")

if __name__ == "__main__":
    main()
