#!/usr/bin/env python3
import pandas as pd
import difflib
from collections import Counter

def main():
    # Load the combined test-set CSV
    df = pd.read_csv("combined_errors.csv", encoding="utf-8")

    counts = Counter()
    for _, row in df.iterrows():
        gt   = str(row["gt"])
        pred = str(row["crnn_pred"])
        sm = difflib.SequenceMatcher(None, gt, pred)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                # Pair up characters in the overlapping spans
                for a, b in zip(gt[i1:i2], pred[j1:j2]):
                    # Skip if GT char is blank or whitespace
                    if a.strip() == "":
                        continue
                    counts[f"{a}->{b}"] += 1
            # we ignore 'insert' and 'delete' tags here

    # Take top 5 real char→char confusion pairs
    top5 = counts.most_common(5)
    print("Top 5 CRNN character confusions (no blanks):")
    for pair, freq in top5:
        print(f"  {pair}: {freq}")

    # Save to CSV
    with open("error_breakdown.csv", "w", encoding="utf-8") as f:
        f.write("true->pred,frequency\n")
        for pair, freq in top5:
            f.write(f"{pair},{freq}\n")
    print("\nSaved to error_breakdown.csv")

if __name__ == "__main__":
    main()
