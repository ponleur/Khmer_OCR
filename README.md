# Khmer OCR

A compact end-to-end OCR system for printed Khmer text using a Convolutional Recurrent Neural Network trained with Connectionist Temporal Classification (CRNN+CTC), with a comparative baseline using Tesseract 5’s Khmer engine.

## Table of Contents

* [Project Structure](#project-structure)
* [Installation & Requirements](#installation--requirements)
* [Data Preparation](#data-preparation)
* [Training the CRNN](#training-the-crnn)
* [Evaluation & Baseline](#evaluation--baseline)
* [Scripts](#scripts)
* [Google Colab](#google-colab)
* [Project Source & Data](#project-source--data)
* [License](#license)

---

## Project Structure

```plaintext
Khmer_OCR/
├── data/
│   └── processed/
│       ├── train/            # Training dataset shards
│       ├── validation/       # Validation dataset shards
│       ├── test/             # Test dataset shards
│       └── vocab.json        # Character vocabulary mapping
├── outputs/
│   ├── checkpoints/          # Saved model weights (.pth)
│   └── logs/                 # TensorBoard logs
├── scripts/
│   ├── data_prep.py          # Prepare data: filter Khmer-only samples, build vocab
│   ├── export_test_images.py # Export test images for Tesseract baseline
│   ├── compute_tesseract.py  # Compute CER/WER for Tesseract outputs
│   └── error_breakdown.py    # Generate confusion and error breakdowns
├── src/
│   ├── dataset.py            # OCRDataset and DataLoader definitions
│   ├── model.py              # CRNN+CTC model definition
│   ├── train.py              # Training loop, checkpointing, TensorBoard
│   └── eval.py               # Evaluation script for CRNN and Tesseract
├── sample_errors.csv         # Sample CRNN errors (first 50)  
├── combined_errors.csv       # Combined CRNN vs. Tesseract errors  
├── Kh_OCR.ipynb              # Google Colab notebook  
├── requirements.txt          # Python dependencies  
└── README.md                 # This file
```

---

## Installation & Requirements

1. Clone the repository and enter the directory:

   ```bash
   git clone https://github.com/ponleur/Khmer_OCR.git && cd Khmer_OCR
   ```

2. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**requirements.txt** includes:

```text
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.0.0
Pillow>=9.0.0
tensorboard>=2.0.0
```

---

## Data Preparation

Prepare the printed Khmer dataset from Hugging Face and split into train/validation/test:

```bash
python scripts/data_prep.py \
  --dataset SoyVitou/62k-images-khmer-printed-dataset \
  --output_dir data/processed
```

This generates:

* `data/processed/vocab.json`
* `data/processed/train/` (≈43 927 samples)
* `data/processed/validation/` (≈5 491 samples)
* `data/processed/test/` (≈5 491 samples)

---

## Training the CRNN

Train the CRNN+CTC model for 20 epochs:

```bash
python src/train.py \
  --data_dir data/processed \
  --epochs 20 \
  --batch_size 16 \
  --num_workers 2 \
  --lr 5e-5 \
  --checkpoint_dir outputs/checkpoints \
  --log_dir outputs/logs
```

Monitor training and validation CER/WER via TensorBoard:

```bash
tensorboard --logdir outputs/logs
# Visit http://localhost:6006
```

---

## Evaluation & Baseline

1. **Export test images** for Tesseract OCR:

   ```bash
   python scripts/export_test_images.py \
     --data_dir data/processed \
     --output_dir tesseract_images
   ```

2. **Run Tesseract** (Khmer) on exported PNGs:

   ```bash
   mkdir -p tesseract_preds
   for img in tesseract_images/*.png; do
     tesseract "$img" stdout -l khm --dpi 300 --psm 7 \
       > tesseract_preds/"$(basename "$img" .png)".txt
   done
   ```

3. **Compute Tesseract CER/WER**:

   ```bash
   python scripts/compute_tesseract.py
   ```

4. **Evaluate CRNN and combine errors**:

   ```bash
   python src/eval.py \
     --data_dir data/processed \
     --checkpoint outputs/checkpoints/crnn_best.pth
   ```

This produces:

```
CRNN Test CER: 0.3376, WER: 0.5459
Tesseract CER: 0.2461, WER: 0.5857
→ combined_errors.csv
→ sample_errors.csv
```

---

## Google Colab

A ready-to-run notebook is available for one-click execution on Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11XKtvHA7hIt7OvkLgN0lLK253-0quopl?usp=sharing)

Mount your Drive and set `data/processed` and `outputs/` paths accordingly.

---

## Project Source & Data

All code, data, and pretrained checkpoints are available on Google Drive:

https://drive.google.com/drive/folders/1MtdxMJCfXv6V8UZgKQDivLCMM7PUgS76?usp=sharing