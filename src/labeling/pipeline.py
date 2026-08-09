"""Orchestration : segmentation → aspects zero-shot → polarité SST-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .aspects import ASPECT_CANDIDATES, ASPECT_HYPOTHESIS_TEMPLATE
from .classifiers import (
    assign_aspects_batch,
    assign_sentiment_batch,
    build_sentiment_pipeline,
    build_zero_shot_pipeline,
)
from .preprocess import iter_review_sentences, load_nlp


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterator[dict] | list[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _max_review_idx(path: Path) -> int:
    """Dernier review_idx déjà écrit (−1 si fichier absent/vide)."""
    if not path.exists() or path.stat().st_size == 0:
        return -1
    last = -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = max(last, int(json.loads(line)["review_idx"]))
    return last


def annotate_sentences(
    sentence_rows: list[dict],
    zs_pipe: Any,
    sent_pipe: Any,
    aspect_threshold: float = 0.50,
    neutral_threshold: float = 0.70,
    top_k_aspects: int = 2,
    zs_batch_size: int = 4,
    sent_batch_size: int = 32,
    chunk_size: int = 32,
    append_path: Path | None = None,
) -> list[dict]:
    """Produit des pseudo-labels aspect-level (une ligne = phrase × aspect).

    Si append_path est fourni, écrit en streaming (reprise possible).
    """
    labeled: list[dict] = []
    n = len(sentence_rows)
    out_f = None
    if append_path is not None:
        append_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(append_path, "a", encoding="utf-8")

    try:
        for start in tqdm(range(0, n, chunk_size), desc="Annotation ABSA"):
            chunk = sentence_rows[start : start + chunk_size]
            texts = [r["sentence"] for r in chunk]

            aspects_per_sent = assign_aspects_batch(
                zs_pipe,
                texts,
                candidate_labels=ASPECT_CANDIDATES,
                hypothesis_template=ASPECT_HYPOTHESIS_TEMPLATE,
                multi_label=True,
                threshold=aspect_threshold,
                top_k=top_k_aspects,
                batch_size=zs_batch_size,
            )
            sentiments = assign_sentiment_batch(
                sent_pipe,
                texts,
                neutral_threshold=neutral_threshold,
                batch_size=sent_batch_size,
            )

            for row, aspects, sentiment in zip(chunk, aspects_per_sent, sentiments):
                for asp in aspects:
                    if float(asp["aspect_score"]) < 0.7 : # aspect_threshold:
                        continue
                    item = {
                        "review_idx": row["review_idx"],
                        "sentence_idx": row["sentence_idx"],
                        "asin": row["asin"],
                        "parent_asin": row["parent_asin"],
                        "user_id": row["user_id"],
                        "rating": row["rating"],
                        "title": row["title"],
                        "verified_purchase": row["verified_purchase"],
                        "review_text": row["review_text"],
                        "sentence": row["sentence"],
                        "aspect": asp["aspect"],
                        "aspect_score": asp["aspect_score"],
                        "aspect_weak": asp.get("weak", False),
                        "sentiment": sentiment["sentiment"],
                        "sentiment_score": sentiment["sentiment_score"],
                        "sentiment_raw_label": sentiment["sentiment_raw_label"],
                    }
                    labeled.append(item)
                    if out_f is not None:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if out_f is not None:
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()

    return labeled


def group_by_review(flat_labels: list[dict]) -> list[dict]:
    """Regroupe les labels plats au format nested review → sentences → aspects."""
    reviews: dict[int, dict] = {}
    for row in flat_labels:
        rid = row["review_idx"]
        if rid not in reviews:
            reviews[rid] = {
                "review_idx": rid,
                "asin": row["asin"],
                "parent_asin": row["parent_asin"],
                "user_id": row["user_id"],
                "rating": row["rating"],
                "title": row["title"],
                "verified_purchase": row["verified_purchase"],
                "text": row["review_text"],
                "sentences": {},
            }
        sents = reviews[rid]["sentences"]
        sid = row["sentence_idx"]
        if sid not in sents:
            sents[sid] = {
                "sentence_idx": sid,
                "text": row["sentence"],
                "sentiment": row["sentiment"],
                "sentiment_score": row["sentiment_score"],
                "aspects": [],
            }
        sents[sid]["aspects"].append(
            {
                "aspect": row["aspect"],
                "aspect_score": row["aspect_score"],
                "aspect_weak": row["aspect_weak"],
                "sentiment": row["sentiment"],
                "sentiment_score": row["sentiment_score"],
            }
        )

    out: list[dict] = []
    for rid in sorted(reviews):
        rev = reviews[rid]
        rev["sentences"] = [
            rev["sentences"][sid] for sid in sorted(rev["sentences"])
        ]
        out.append(rev)
    return out


def run_labeling(
    input_path: str | Path,
    output_flat: str | Path,
    output_nested: str | Path | None = None,
    limit: int | None = None,
    zs_model: str = "typeform/distilbert-base-uncased-mnli",
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    aspect_threshold: float = 0.50,
    neutral_threshold: float = 0.70,
    top_k_aspects: int = 2,
    zs_batch_size: int = 4,
    sent_batch_size: int = 32,
    chunk_size: int = 32,
    spacy_model: str = "en_core_web_sm",
    resume: bool = True,
) -> dict[str, int]:
    """Pipeline de bout en bout : JSONL avis → JSONL labels aspect-level."""
    output_flat = Path(output_flat)
    reviews = load_jsonl(input_path, limit=limit)

    start_review = 0
    if resume and output_flat.exists() and output_flat.stat().st_size > 0:
        last = _max_review_idx(output_flat)
        start_review = last + 1
        print(f"Resume: {start_review} avis deja labels, reprise a l'index {start_review}")

    reviews_todo = [
        (idx, rev) for idx, rev in enumerate(reviews) if idx >= start_review
    ]
    if not reviews_todo:
        print("Rien a annoter (deja a jour).")
        flat_all = load_jsonl(output_flat)
        n_nested = 0
        if output_nested is not None:
            nested = group_by_review(flat_all)
            n_nested = write_jsonl(output_nested, nested)
        return {
            "n_reviews": len(reviews),
            "n_sentences": 0,
            "n_aspect_labels": len(flat_all),
            "n_nested_reviews": n_nested,
        }

    nlp = load_nlp(spacy_model)
    # Ré-indexation : garder le review_idx global d'origine
    sentence_rows = []
    for idx, rev in reviews_todo:
        rows = iter_review_sentences(nlp, [rev])
        for r in rows:
            r["review_idx"] = idx
            sentence_rows.append(r)

    zs_pipe = build_zero_shot_pipeline(model_name=zs_model)
    sent_pipe = build_sentiment_pipeline(model_name=sentiment_model)

    # Mode append si reprise, sinon fichier neuf
    if start_review == 0 and output_flat.exists():
        output_flat.unlink()

    flat_new = annotate_sentences(
        sentence_rows,
        zs_pipe=zs_pipe,
        sent_pipe=sent_pipe,
        aspect_threshold=aspect_threshold,
        neutral_threshold=neutral_threshold,
        top_k_aspects=top_k_aspects,
        zs_batch_size=zs_batch_size,
        sent_batch_size=sent_batch_size,
        chunk_size=chunk_size,
        append_path=output_flat,
    )

    flat_all = load_jsonl(output_flat)
    n_nested = 0
    if output_nested is not None:
        nested = group_by_review(flat_all)
        n_nested = write_jsonl(output_nested, nested)

    return {
        "n_reviews": len(reviews),
        "n_sentences": len(sentence_rows),
        "n_aspect_labels_new": len(flat_new),
        "n_aspect_labels_total": len(flat_all),
        "n_nested_reviews": n_nested,
        "resumed_from_review_idx": start_review,
    }
