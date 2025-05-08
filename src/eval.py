#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import json
import csv
import torch
from datasets import load_from_disk

# Ensure src/ is on the import path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
)
from dataset import get_dataloader
from model import CRNN
from utils import ctc_greedy_decoder, cer, wer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CRNN and compare with Tesseract predictions")
    p.add_argument("--data_dir",    required=True,
                   help="Path to processed data directory (contains test split and vocab.json)")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to the CRNN checkpoint to load (best_crnn.pth)")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--tess_dir",    default="tesseract_preds",
                   help="Directory containing Tesseract .txt predictions")
    p.add_argument("--combined_csv", default="combined_errors.csv",
                   help="Output CSV with GT, CRNN, and Tesseract predictions for all test samples")
    p.add_argument("--sample_csv",   default="sample_errors.csv",
                   help="Output CSV with up to 50 CRNN errors (with Tesseract column)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path, encoding="utf-8") as vf:
        vd = json.load(vf)
    idx_to_char = {i+1: ch for i, ch in enumerate(vd["vocab"]) }
    idx_to_char[0] = ""  # CTC blank
    num_classes = len(vd["vocab"] )

    # Load CRNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    # Load test dataset for ground-truth
    test_ds = load_from_disk(os.path.join(args.data_dir, "test"))
    gt_list = [ex["text"] for ex in test_ds]

    # Create DataLoader for CRNN predictions
    test_loader = get_dataloader(
        "test", args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Predict with CRNN
    all_crnn = []
    with torch.no_grad():
        for imgs, labs, lengths in test_loader:
            imgs = imgs.to(device)
            out  = model(imgs)  # (T, B, C)
            preds = ctc_greedy_decoder(out, idx_to_char)
            all_crnn.extend(preds)

    # Load Tesseract predictions
    tess_files = sorted(glob.glob(os.path.join(args.tess_dir, "*.txt")))
    tess_basenames = [os.path.splitext(os.path.basename(f))[0] for f in tess_files]
    all_tess = [open(f, encoding="utf-8").read().strip() for f in tess_files]

    # Compute metrics for CRNN alone
    test_cer = cer(all_crnn, gt_list)
    test_wer = wer(all_crnn, gt_list)
    print(f"CRNN Test CER: {test_cer:.4f}, WER: {test_wer:.4f}")

    # Compute metrics for Tesseract baseline
    tess_cer = cer(all_tess, gt_list)
    tess_wer = wer(all_tess, gt_list)
    print(f"Tesseract CER: {tess_cer:.4f}, WER: {tess_wer:.4f}")

    # Write combined CSV for all samples
    with open(args.combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt", "crnn_pred", "tess_pred"])
        for fn, gt, cr, te in zip(tess_basenames, gt_list, all_crnn, all_tess):
            writer.writerow([fn, gt, cr, te])
    print(f"→ Combined CSV written to {args.combined_csv}")

    # Write sample errors CSV (up to 50 CRNN errors)
    count = 0
    with open(args.sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt", "crnn_pred", "tess_pred"])
        for fn, gt, cr, te in zip(tess_basenames, gt_list, all_crnn, all_tess):
            if gt != cr:
                writer.writerow([fn, gt, cr, te])
                count += 1
                if count >= 50:
                    break
    print(f"→ Sample errors CSV written to {args.sample_csv} (first {count} mismatches)")

if __name__ == "__main__":
    main()
