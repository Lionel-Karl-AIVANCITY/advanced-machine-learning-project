#!/usr/bin/env python
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

nested_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "data/raw/smoke_aspect_labels.jsonl"
)
flat_path = Path(
    sys.argv[2] if len(sys.argv) > 2 else "data/raw/smoke_aspect_labels_flat.jsonl"
)

with open(nested_path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        print("=== REVIEW", r["review_idx"], "rating", r["rating"], "===")
        print("TEXT:", r["text"][:180].replace("\n", " "), "...")
        for s in r["sentences"]:
            print(f"  [{s['sentence_idx']}] {s['text'][:120]}")
            print(f"       sent={s['sentiment']} ({s['sentiment_score']:.2f})")
            for a in s["aspects"]:
                print(
                    f"       - {a['aspect']}: {a['aspect_score']:.2f} weak={a['aspect_weak']}"
                )
        print()
        if i >= 2:
            break

asp = Counter()
pol = Counter()
n = 0
with open(flat_path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        n += 1
        asp[row["aspect"]] += 1
        pol[row["sentiment"]] += 1
print("n_flat", n)
print("aspects", asp.most_common())
print("polarity", dict(pol))
