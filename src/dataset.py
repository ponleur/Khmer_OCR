import os, json
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_vocab(path):
    d = json.load(open(path,encoding="utf-8"))
    return d["char_to_idx"]

class OCRDataset(Dataset):
    def __init__(self, split_dir, vocab_path, transform=None):
        self.ds = load_from_disk(split_dir)
        self.c2i = load_vocab(vocab_path)
        self.tf = transform or transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((32,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        img = self.tf(ex["image"])
        lab = torch.tensor([self.c2i[ch] for ch in ex["text"]],dtype=torch.long)
        return img, lab

def collate_fn(batch):
    imgs, labs = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = torch.tensor([len(l) for l in labs],dtype=torch.long)
    labs_cat = torch.cat(labs)
    return imgs, labs_cat, lengths

def get_dataloader(split, data_dir="data/processed",
                   batch_size=32, shuffle=False, num_workers=4):
    ds = OCRDataset(f"{data_dir}/{split}",
                    f"{data_dir}/vocab.json")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)
