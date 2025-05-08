import os, json, argparse
from datasets import load_dataset, DatasetDict

def is_khmer(text: str) -> bool:
    return any('\u1780' <= ch <= '\u17FF' for ch in text)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True)
    p.add_argument("--output_dir",default="data/processed")
    args = p.parse_args()

    raw = load_dataset(args.dataset, split="train")
    raw = raw.filter(lambda ex: is_khmer(ex["text"]))
    print(f"→ Khmer-only samples: {len(raw)}")

    # 80/10/10 train/val/test
    splits = raw.train_test_split(test_size=0.2, seed=42)
    vt     = splits["test"].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train":      splits["train"],
        "validation": vt["train"],
        "test":       vt["test"],
    })

    # build vocab
    chars = set()
    for split in dataset.values():
        for ex in split:
            chars.update(ex["text"])
    vocab = sorted(chars)
    char_to_idx = {ch: i+1 for i,ch in enumerate(vocab)}  # 0 = blank

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/vocab.json","w",encoding="utf-8") as f:
        json.dump({"vocab":vocab,"char_to_idx":char_to_idx},
                  f, ensure_ascii=False, indent=2)
    print(f"→ Saved vocab ({len(vocab)}) to {args.output_dir}/vocab.json")

    for name, ds in dataset.items():
        out = f"{args.output_dir}/{name}"
        ds.save_to_disk(out)
        print(f"→ Saved {name} ({len(ds)}) to {out}")

if __name__=="__main__":
    main()