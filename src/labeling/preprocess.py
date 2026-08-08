"""Nettoyage HTML et segmentation en phrases (spaCy)."""

from __future__ import annotations

import html
import re
from typing import Iterable

import spacy
from spacy.language import Language

_WS_RE = re.compile(r"\s+")


def load_nlp(model_name: str = "en_core_web_sm") -> Language:
    """Charge spaCy en mode segmentation seule (rapide)."""
    nlp = spacy.load(model_name, disable=["ner", "lemmatizer", "attribute_ruler"])
    return nlp


def clean_text(text: str | None) -> str:
    """Décode les entités HTML et normalise les espaces."""
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def segment_sentences(nlp: Language, text: str, min_chars: int = 8) -> list[str]:
    """Découpe un avis en phrases ; ignore les fragments trop courts."""
    text = clean_text(text)
    if not text:
        return []
    doc = nlp(text)
    sentences: list[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) >= min_chars:
            sentences.append(s)
    # Fallback : avis sans ponctuation claire → une seule unité
    if not sentences and len(text) >= min_chars:
        sentences = [text]
    return sentences


def iter_review_sentences(
    nlp: Language,
    reviews: Iterable[dict],
    text_field: str = "text",
    min_chars: int = 8,
) -> list[dict]:
    """Produit une ligne par phrase avec métadonnées de l'avis source."""
    rows: list[dict] = []
    for review_idx, review in enumerate(reviews):
        raw = review.get(text_field) or ""
        for sent_idx, sentence in enumerate(
            segment_sentences(nlp, raw, min_chars=min_chars)
        ):
            rows.append(
                {
                    "review_idx": review_idx,
                    "sentence_idx": sent_idx,
                    "sentence": sentence,
                    "asin": review.get("asin"),
                    "parent_asin": review.get("parent_asin"),
                    "user_id": review.get("user_id"),
                    "rating": review.get("rating"),
                    "title": review.get("title"),
                    "verified_purchase": review.get("verified_purchase"),
                    "review_text": clean_text(raw),
                }
            )
    return rows
