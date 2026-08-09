#!/usr/bin/env python
"""Construit des pseudo-labels ABSA aspect-level via zero-shot NLI + SST-2.

Exemple (smoke test) :
    python scripts/build_aspect_labels.py --limit 50

Exemple (jeu complet, CPU-friendly) :
    python scripts/build_aspect_labels.py

Exemple (meilleure qualite NLI, plus lent) :
    python scripts/build_aspect_labels.py --zs-model facebook/bart-large-mnli
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.pipeline import run_labeling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pseudo-annotation ABSA : spaCy + zero-shot aspects + sentiment SST-2"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "Data" / "Baby_Products_echanillon.jsonl",
        help="JSONL d'avis bruts",
    )
    p.add_argument(
        "--output-flat",
        type=Path,
        default=ROOT / "Data" / "Baby_Products_aspect_labels_flat.jsonl",
        help="Sortie plate (1 ligne = phrase x aspect)",
    )
    p.add_argument(
        "--output-nested",
        type=Path,
        default=ROOT / "Data" / "Baby_Products_aspect_labels.jsonl",
        help="Sortie nested (review -> sentences -> aspects)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max d'avis a traiter (None = tous)",
    )
    p.add_argument(
        "--zs-model",
        default="typeform/distilbert-base-uncased-mnli",
        help=(
            "Modele zero-shot NLI. Defaut DistilBERT-MNLI (CPU). "
            "Pour plus de qualite: facebook/bart-large-mnli"
        ),
    )
    p.add_argument(
        "--sentiment-model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Classifieur de sentiment pre-entraine",
    )
    p.add_argument(
        "--aspect-threshold",
        type=float,
        default=0.50,
        help="Seuil multi-label pour garder un aspect",
    )
    p.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.70,
        help="Sous ce score SST-2 -> polarite 'neutral'",
    )
    p.add_argument("--top-k-aspects", type=int, default=2)
    p.add_argument("--zs-batch-size", type=int, default=4)
    p.add_argument("--sent-batch-size", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore le fichier flat existant et recommence a zero",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Pseudo-annotation ABSA ===")
    print(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )

    stats = run_labeling(
        input_path=args.input,
        output_flat=args.output_flat,
        output_nested=args.output_nested,
        limit=args.limit,
        zs_model=args.zs_model,
        sentiment_model=args.sentiment_model,
        aspect_threshold=args.aspect_threshold,
        neutral_threshold=args.neutral_threshold,
        top_k_aspects=args.top_k_aspects,
        zs_batch_size=args.zs_batch_size,
        sent_batch_size=args.sent_batch_size,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )
    print("=== Termine ===")
    print(json.dumps(stats, indent=2))
    print(f"Flat   -> {args.output_flat}")
    print(f"Nested -> {args.output_nested}")


if __name__ == "__main__":
    main()
