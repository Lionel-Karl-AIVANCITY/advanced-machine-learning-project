"""
Demo Streamlit — Aspect-Based Sentiment Analysis (BERT fine-tune).

Lancement (depuis la racine du repo) :
    streamlit run app.py

Ce que le modele sait faire : predire le sentiment (positif/negatif) pour
UN aspect donne dans UNE phrase donnee. Il ne detecte pas tout seul quels
aspects sont mentionnes — l'utilisateur les choisit via la barre laterale.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
BERT_DIR = REPO_ROOT / "bert_outputs"
DATA_DIR = REPO_ROOT / "data_splits"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fallback si data_splits/train.jsonl est absent
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from labeling.aspects import ASPECT_CANDIDATES
except Exception:  # pragma: no cover
    ASPECT_CANDIDATES = [
        "product quality and performance",
        "price, cost and value for money",
        "design, appearance and looks",
        "size, dimensions and fit",
        "comfort, softness and texture",
        "durability, washability and longevity",
        "safety for babies and children",
        "ease of use, assembly and convenience",
        "packaging, shipping and delivery",
        "customer service and seller support",
    ]

st.set_page_config(
    page_title="ABSA — Avis Baby Products",
    page_icon="🍼",
    layout="wide",
)

SENTIMENT_STYLE = {
    "positive": ("🟢", "Positif"),
    "negative": ("🔴", "Negatif"),
}


@st.cache_resource
def load_model():
    ckpt_path = BERT_DIR / "best_model_state.pt"
    tok_dir = BERT_DIR / "tokenizer"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable: {ckpt_path}. "
            "Entrainement requis (scripts/data_train/train_bert.py)."
        )
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer introuvable: {tok_dir}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint["model_name"], num_labels=len(checkpoint["label2id"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    id2label = {v: k for k, v in checkpoint["label2id"].items()}
    return model, tokenizer, id2label, checkpoint["max_len"]


@st.cache_data
def load_aspect_taxonomy() -> list[str]:
    """Liste des aspects depuis le train set, sinon taxonomie du projet."""
    train_path = DATA_DIR / "train.jsonl"
    if not train_path.exists():
        return list(ASPECT_CANDIDATES)

    aspects: set[str] = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            aspects.add(json.loads(line)["aspect"])
    return sorted(aspects) if aspects else list(ASPECT_CANDIDATES)


def split_sentences(text: str) -> list[str]:
    """Segmentation legere par ponctuation forte (. ! ?)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


@torch.no_grad()
def predict_batch(model, tokenizer, id2label, max_len, pairs):
    """pairs : liste (aspect, phrase) -> liste (label, confiance)."""
    aspects = [p[0] for p in pairs]
    sentences = [p[1] for p in pairs]

    # Meme format que train_bert.py : [CLS] aspect [SEP] phrase [SEP]
    encodings = tokenizer(
        text=aspects,
        text_pair=sentences,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    logits = model(**encodings).logits
    probs = F.softmax(logits, dim=1)
    confidences, pred_ids = probs.max(dim=1)

    return [
        (id2label[pred_id], conf)
        for pred_id, conf in zip(
            pred_ids.cpu().tolist(), confidences.cpu().tolist()
        )
    ]


def build_summary(rows, selected_aspects):
    """Agrege les resultats phrase par phrase en un verdict par aspect."""
    summary = []
    for aspect in selected_aspects:
        aspect_rows = [r for r in rows if r["Aspect"] == aspect]
        labels = [r["Sentiment"] for r in aspect_rows]
        counts = Counter(labels)

        if len(counts) == 1:
            dominant = labels[0]
            avg_conf = sum(r["Confiance"] for r in aspect_rows) / len(aspect_rows)
            emoji, label_fr = SENTIMENT_STYLE.get(dominant, ("⚪", dominant))
            summary.append(
                {
                    "Aspect": aspect,
                    "Verdict": f"{emoji} {label_fr}",
                    "Detail": (
                        f"{len(aspect_rows)} phrase(s) — "
                        f"confiance moy. {avg_conf * 100:.0f}%"
                    ),
                }
            )
        else:
            detail = ", ".join(
                f"{v}x {SENTIMENT_STYLE.get(k, ('', k))[1]}" for k, v in counts.items()
            )
            summary.append(
                {"Aspect": aspect, "Verdict": "Mixte", "Detail": detail}
            )

    return pd.DataFrame(summary)


def main():
    st.title("Analyse d'avis — Aspect-Based Sentiment Analysis")
    st.caption(
        "Modele BERT fine-tune sur des avis Amazon, categorie Baby Products"
    )

    try:
        model, tokenizer, id2label, max_len = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    all_aspects = load_aspect_taxonomy()

    with st.sidebar:
        st.header("Parametres")
        st.caption(f"Device: {DEVICE}")
        st.caption(f"BERT_DIR: {BERT_DIR}")
        selected_aspects = st.multiselect(
            "Aspects a evaluer",
            options=all_aspects,
            default=all_aspects[: min(4, len(all_aspects))],
        )
        st.caption(f"{len(all_aspects)} aspects disponibles")
        st.info(
            "Le modele predit le sentiment pour CHAQUE aspect selectionne, "
            "meme s'il n'est pas vraiment mentionne dans la phrase. "
            "Selectionnez uniquement les aspects pertinents."
        )

    review_text = st.text_area(
        "Collez un avis produit (une ou plusieurs phrases)",
        height=150,
        placeholder=(
            "Ex: The plate is super cute but the suction doesn't hold at all "
            "after a few uses. Price is fair for the quality though."
        ),
    )

    run = st.button("Analyser", type="primary")
    if not run:
        return

    if not review_text.strip():
        st.warning("Merci de saisir un avis avant de lancer l'analyse.")
        return
    if not selected_aspects:
        st.warning("Selectionnez au moins un aspect dans la barre laterale.")
        return

    sentences = split_sentences(review_text)
    if not sentences:
        st.warning("Impossible de detecter des phrases dans le texte saisi.")
        return

    pairs = [
        (aspect, sentence)
        for sentence in sentences
        for aspect in selected_aspects
    ]

    with st.spinner(
        f"Analyse de {len(sentences)} phrase(s) x {len(selected_aspects)} aspect(s)..."
    ):
        predictions = predict_batch(
            model, tokenizer, id2label, max_len, pairs
        )

    rows = []
    idx = 0
    for sentence in sentences:
        for aspect in selected_aspects:
            label, confidence = predictions[idx]
            idx += 1
            rows.append(
                {
                    "Phrase": sentence,
                    "Aspect": aspect,
                    "Sentiment": label,
                    "Confiance": confidence,
                }
            )

    st.subheader("Verdict par aspect")
    summary_df = build_summary(rows, selected_aspects)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Detail phrase par phrase")
    for sentence in sentences:
        st.markdown(f"**« {sentence} »**")
        # Evite trop de colonnes si beaucoup d'aspects selectionnes
        n_cols = min(4, len(selected_aspects))
        for start in range(0, len(selected_aspects), n_cols):
            chunk = selected_aspects[start : start + n_cols]
            cols = st.columns(len(chunk))
            for col, aspect in zip(cols, chunk):
                match = next(
                    r
                    for r in rows
                    if r["Phrase"] == sentence and r["Aspect"] == aspect
                )
                emoji, label_fr = SENTIMENT_STYLE.get(
                    match["Sentiment"], ("⚪", match["Sentiment"])
                )
                with col:
                    st.metric(
                        label=aspect,
                        value=f"{emoji} {label_fr}",
                        delta=f"{match['Confiance'] * 100:.0f}% confiance",
                        delta_color="off",
                    )
        st.divider()


if __name__ == "__main__":
    main()
