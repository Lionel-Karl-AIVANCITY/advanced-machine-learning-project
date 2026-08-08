"""Classifieurs HF : zero-shot NLI (aspects) + sentiment SST-2 (polarité)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import pipeline

from .aspects import (
    ASPECT_CANDIDATES,
    ASPECT_HYPOTHESIS_TEMPLATE,
    FALLBACK_ASPECT,
    lexical_aspect_hits,
)


def get_device(prefer_cuda: bool = True) -> int | str:
    if prefer_cuda and torch.cuda.is_available():
        return 0
    return -1


def build_zero_shot_pipeline(
    model_name: str = "facebook/bart-large-mnli",
    device: int | str | None = None,
) -> Any:
    """Pipeline HF zero-shot-classification (NLI)."""
    if device is None:
        device = get_device()
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )


def build_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: int | str | None = None,
) -> Any:
    """Classifieur de sentiment pré-entraîné (SST-2)."""
    if device is None:
        device = get_device()
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=512,
    )


def assign_aspects(
    zs_pipe: Any,
    sentence: str,
    candidate_labels: list[str] | None = None,
    hypothesis_template: str = ASPECT_HYPOTHESIS_TEMPLATE,
    multi_label: bool = True,
    threshold: float = 0.45,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    """Assigne un ou plusieurs aspects via zero-shot NLI.

    Returns:
        Liste de dicts {aspect, score}, éventuellement FALLBACK_ASPECT.
    """
    labels = candidate_labels or ASPECT_CANDIDATES
    result = zs_pipe(
        sentence,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
    )
    pairs = list(zip(result["labels"], result["scores"]))
    selected = [(lab, float(sc)) for lab, sc in pairs if float(sc) >= threshold]
    selected.sort(key=lambda x: x[1], reverse=True)
    selected = selected[:top_k]

    if not selected:
        # Meilleur aspect même sous le seuil, étiqueté comme fallback soft
        best_lab, best_sc = pairs[0]
        return [{"aspect": best_lab, "aspect_score": float(best_sc), "weak": True}]

    return [
        {"aspect": lab, "aspect_score": sc, "weak": False} for lab, sc in selected
    ]


def assign_aspects_batch(
    zs_pipe: Any,
    sentences: list[str],
    candidate_labels: list[str] | None = None,
    hypothesis_template: str = ASPECT_HYPOTHESIS_TEMPLATE,
    multi_label: bool = True,
    threshold: float = 0.45,
    top_k: int = 3,
    batch_size: int = 8,
) -> list[list[dict[str, float | str | bool]]]:
    """Version batch du zero-shot aspect."""
    labels = candidate_labels or ASPECT_CANDIDATES
    outputs = zs_pipe(
        sentences,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
        batch_size=batch_size,
    )
    # pipeline peut renvoyer un dict unique si len==1
    if isinstance(outputs, dict):
        outputs = [outputs]

    all_aspects: list[list[dict[str, float | str | bool]]] = []
    for sentence, result in zip(sentences, outputs):
        score_map = {
            lab: float(sc) for lab, sc in zip(result["labels"], result["scores"])
        }
        selected = [(lab, sc) for lab, sc in score_map.items() if sc >= threshold]
        selected.sort(key=lambda x: x[1], reverse=True)
        selected = selected[:top_k]

        # Weak supervision : ajouter les aspects à fort signal lexical
        chosen = {lab for lab, _ in selected}
        for cue_aspect in lexical_aspect_hits(sentence):
            if cue_aspect in chosen:
                continue
            cue_score = max(score_map.get(cue_aspect, 0.0), threshold)
            selected.append((cue_aspect, cue_score))
            chosen.add(cue_aspect)

        if not selected:
            best_lab = max(score_map, key=score_map.get)
            all_aspects.append(
                [
                    {
                        "aspect": best_lab,
                        "aspect_score": score_map[best_lab],
                        "weak": True,
                    }
                ]
            )
        else:
            selected.sort(key=lambda x: x[1], reverse=True)
            all_aspects.append(
                [
                    {
                        "aspect": lab,
                        "aspect_score": sc,
                        "weak": sc < threshold,
                    }
                    for lab, sc in selected
                ]
            )
    return all_aspects


def score_to_polarity(
    label: str,
    score: float,
    neutral_threshold: float = 0.70,
) -> tuple[str, float]:
    """Mappe SST-2 (POSITIVE/NEGATIVE) vers positif / négatif / neutre.

    Si la confiance est faible (< neutral_threshold), on assigne 'neutral'.
    """
    lab = label.upper()
    if score < neutral_threshold:
        return "neutral", float(score)
    if lab in {"POSITIVE", "POS", "LABEL_1"}:
        return "positive", float(score)
    if lab in {"NEGATIVE", "NEG", "LABEL_0"}:
        return "negative", float(score)
    return "neutral", float(score)


def assign_sentiment(
    sent_pipe: Any,
    sentence: str,
    neutral_threshold: float = 0.70,
) -> dict[str, float | str]:
    """Pseudo-label de polarité pour une phrase."""
    out = sent_pipe(sentence)[0]
    polarity, conf = score_to_polarity(
        out["label"], float(out["score"]), neutral_threshold=neutral_threshold
    )
    return {
        "sentiment": polarity,
        "sentiment_score": conf,
        "sentiment_raw_label": out["label"],
    }


def assign_sentiment_batch(
    sent_pipe: Any,
    sentences: list[str],
    neutral_threshold: float = 0.70,
    batch_size: int = 32,
) -> list[dict[str, float | str]]:
    """Version batch du classifieur de sentiment."""
    outputs = sent_pipe(sentences, batch_size=batch_size)
    results: list[dict[str, float | str]] = []
    for out in outputs:
        polarity, conf = score_to_polarity(
            out["label"], float(out["score"]), neutral_threshold=neutral_threshold
        )
        results.append(
            {
                "sentiment": polarity,
                "sentiment_score": conf,
                "sentiment_raw_label": out["label"],
            }
        )
    return results


# Réexport pour clarté API
__all__ = [
    "FALLBACK_ASPECT",
    "build_zero_shot_pipeline",
    "build_sentiment_pipeline",
    "assign_aspects",
    "assign_aspects_batch",
    "assign_sentiment",
    "assign_sentiment_batch",
    "score_to_polarity",
]
