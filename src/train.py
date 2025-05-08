#!/usr/bin/env python3
import os, json, argparse, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from model import CRNN
from utils import ctc_greedy_decoder, cer, wer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="data/processed")
    p.add_argument("--epochs",   type=int,   default=10)
    p.add_argument("--batch_size",type=int,   default=16)
    p.add_argument("--lr",       type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_size",  type=int, default=512,
                   help="RNN hidden size (nh)")
    p.add_argument("--dropout",      type=float, default=0.3,
                   help="Dropout prob between RNN layers")
    p.add_argument("--num_workers",type=int,   default=2)
    p.add_argument("--checkpoint_dir", default="outputs/checkpoints")
    p.add_argument("--log_dir",        default="outputs/logs")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1) Load vocab
    vd = json.load(open(os.path.join(args.data_dir, "vocab.json"), encoding="utf-8"))
    vocab = vd["vocab"]
    idx_to_char = {i+1: ch for i, ch in enumerate(vocab)}
    idx_to_char[0] = ""  # blank
    num_classes = len(vocab)

    # 2) DataLoaders
    train_loader = get_dataloader("train",      args.data_dir,
                                  args.batch_size, True,  args.num_workers)
    val_loader   = get_dataloader("validation", args.data_dir,
                                  args.batch_size, False, args.num_workers)

    # 3) Model + Loss + Optimizer + Scheduler
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = CRNN(num_classes,
                     nh=args.hidden_size,
                     dropout_p=args.dropout).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=1)

    # 4) TensorBoard
    writer   = SummaryWriter(log_dir=args.log_dir)
    best_cer = float('inf')

    # 5) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labs, lengths in train_loader:
            imgs, labs, lengths = imgs.to(device), labs.to(device), lengths.to(device)
            logits = model(imgs)      # (T, B, C)
            T, B, C = logits.size()
            inp_lens = torch.full((B,), T, dtype=torch.long, device=device)
            logp = logits.log_softmax(2)

            loss = criterion(logp, labs, inp_lens, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
            step = (epoch-1)*len(train_loader) + len(train_loader)
            writer.add_scalar("Train/Loss", loss.item(), step)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {avg_loss:.4f}")

        # 6) Checkpoint
        ckpt = os.path.join(args.checkpoint_dir, f"crnn_epoch{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt)

        # 7) Validation
        model.eval()
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for imgs, labs, lengths in val_loader:
                imgs = imgs.to(device)
                out  = model(imgs)                 # (T, B, C)
                preds = ctc_greedy_decoder(out, idx_to_char)

                # rebuild ground-truth strings
                offset = 0
                for L in lengths.tolist():
                    seq = labs[offset:offset+L].tolist()
                    all_tgts.append("".join(idx_to_char[i] for i in seq))
                    offset += L
                all_preds.extend(preds)

        val_cer = cer(all_preds, all_tgts)
        val_wer = wer(all_preds, all_tgts)
        print(f"Epoch {epoch} — Val CER: {val_cer:.4f}, WER: {val_wer:.4f}")

        writer.add_scalar("Val/CER", val_cer, epoch)
        writer.add_scalar("Val/WER", val_wer, epoch)

        # 8) Best‐model & LR scheduler
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_crnn.pth"))
            print(f"→ New best model saved (CER={best_cer:.4f})")
        scheduler.step(val_cer)

    writer.close()

if __name__ == "__main__":
    main()
