#!/usr/bin/env python3
import sys, os
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import glob, json
from utils import cer, wer
from dataset import get_dataloader

# load ground-truth
vd = json.load(open("data/processed/vocab.json", encoding="utf-8"))
gt_list = []
loader = get_dataloader("test", "data/processed",
                        batch_size=128, shuffle=False, num_workers=0)
for _, labs, lengths in loader:
    off = 0
    for L in lengths.tolist():
        seq = labs[off:off+L].tolist()
        gt_list.append("".join(vd["vocab"][i-1] for i in seq))
        off += L

# load Tesseract preds
preds = []
for fn in sorted(glob.glob("tesseract_preds/*.txt")):
    preds.append(open(fn, encoding="utf-8").read().strip())

print("Tesseract CER:", cer(preds, gt_list))
print("Tesseract WER:", wer(preds, gt_list))
