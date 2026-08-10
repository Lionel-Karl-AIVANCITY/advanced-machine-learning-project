"""
Demo Streamlit — ABSA Baby Products (BERT fine-tune).

Objectif business : analyser un corpus d'avis et produire une synthese
par aspect (points forts / axes d'amelioration), pas seulement un avis isole.

Lancement (racine du repo) :
    streamlit run app/app.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = REPO_ROOT / "models" / "bert"
DATA_DIR = REPO_ROOT / "data" / "splits"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from labeling.aspects import ASPECT_CANDIDATES, lexical_aspect_hits
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

    def lexical_aspect_hits(sentence: str) -> list[str]:
        return []


FALLBACK_ASPECT = "product quality and performance"
INFER_BATCH_SIZE = 32

st.set_page_config(
    page_title="ABSA — Synthese avis Baby Products",
    page_icon="🍼",
    layout="wide",
)

SENTIMENT_STYLE = {
    "positive": ("🟢", "Positif"),
    "negative": ("🔴", "Negatif"),
}


# ---------------------------------------------------------------------------
# Chargement modele / taxonomie
# ---------------------------------------------------------------------------
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
    train_path = DATA_DIR / "train.jsonl"
    if not train_path.exists():
        return list(ASPECT_CANDIDATES)
    aspects: set[str] = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            aspects.add(json.loads(line)["aspect"])
    return sorted(aspects) if aspects else list(ASPECT_CANDIDATES)


# ---------------------------------------------------------------------------
# Ingestion avis
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def parse_pasted_reviews(raw: str) -> list[str]:
    """
    Une ligne = un avis (CSV-like).
    Les lignes vides sont ignorees.
    """
    reviews = []
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        if line:
            reviews.append(line)
    return reviews


def parse_uploaded_csv(uploaded_file) -> tuple[list[str], str | None]:
    """
    Charge un CSV et extrait la colonne texte.
    Colonnes reconnues (dans l'ordre) : text, review, review_text, sentence, avis, content.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";")

    if df.empty:
        return [], "Le CSV est vide."

    candidates = [
        "text",
        "review",
        "review_text",
        "sentence",
        "avis",
        "content",
        "body",
        "comment",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        # Si une seule colonne, on la prend
        if len(df.columns) == 1:
            col = df.columns[0]
        else:
            return (
                [],
                "Colonne texte introuvable. Attendu: text / review / review_text / "
                f"sentence / avis. Colonnes vues: {list(df.columns)}",
            )

    reviews = (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s.str.len() > 0]
        .tolist()
    )
    return reviews, None


def detect_aspects_for_sentence(
    sentence: str,
    taxonomy: list[str],
    forced_aspects: list[str],
    auto_detect: bool,
) -> list[str]:
    aspects: list[str] = []
    if auto_detect:
        hits = [a for a in lexical_aspect_hits(sentence) if a in taxonomy]
        aspects.extend(hits)
        if not aspects:
            # Phrase sans cue lexical clair -> aspect qualite generique
            if FALLBACK_ASPECT in taxonomy:
                aspects.append(FALLBACK_ASPECT)
    for aspect in forced_aspects:
        if aspect not in aspects:
            aspects.append(aspect)
    return aspects


# ---------------------------------------------------------------------------
# Inference BERT
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(model, tokenizer, id2label, max_len, pairs, batch_size=INFER_BATCH_SIZE):
    """pairs : (aspect, sentence) -> (label, confiance)."""
    if not pairs:
        return []

    results = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        aspects = [p[0] for p in chunk]
        sentences = [p[1] for p in chunk]
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
        for pred_id, conf in zip(pred_ids.cpu().tolist(), confidences.cpu().tolist()):
            results.append((id2label[pred_id], float(conf)))
    return results


def analyze_corpus(
    reviews: list[str],
    model,
    tokenizer,
    id2label,
    max_len,
    taxonomy: list[str],
    forced_aspects: list[str],
    auto_detect: bool,
) -> pd.DataFrame:
    """Produit un dataframe plat : une ligne = (avis, phrase, aspect, sentiment)."""
    jobs = []  # metadata alinees avec pairs
    pairs = []
    for review_idx, review in enumerate(reviews):
        sentences = split_sentences(review)
        if not sentences:
            sentences = [review.strip()] if review.strip() else []
        for sent_idx, sentence in enumerate(sentences):
            aspects = detect_aspects_for_sentence(
                sentence, taxonomy, forced_aspects, auto_detect
            )
            for aspect in aspects:
                pairs.append((aspect, sentence))
                jobs.append(
                    {
                        "review_idx": review_idx,
                        "sentence_idx": sent_idx,
                        "review": review,
                        "sentence": sentence,
                        "aspect": aspect,
                    }
                )

    preds = predict_batch(model, tokenizer, id2label, max_len, pairs)
    rows = []
    for meta, (label, conf) in zip(jobs, preds):
        rows.append({**meta, "sentiment": label, "confidence": conf})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthese agregée
# ---------------------------------------------------------------------------
def build_aspect_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for aspect, g in df.groupby("aspect"):
        n = len(g)
        n_pos = int((g["sentiment"] == "positive").sum())
        n_neg = int((g["sentiment"] == "negative").sum())
        pct_pos = 100.0 * n_pos / n
        pct_neg = 100.0 * n_neg / n
        # Priorite : volume de negatifs (impact business)
        priority = n_neg
        rows.append(
            {
                "aspect": aspect,
                "mentions": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "pct_positive": round(pct_pos, 1),
                "pct_negative": round(pct_neg, 1),
                "priority_score": priority,
                "avg_confidence": round(float(g["confidence"].mean()), 3),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        by=["priority_score", "pct_negative", "mentions"],
        ascending=[False, False, False],
    )
    return out.reset_index(drop=True)


def pick_verbatims(df: pd.DataFrame, aspect: str, sentiment: str, k: int = 3) -> pd.DataFrame:
    subset = df[(df["aspect"] == aspect) & (df["sentiment"] == sentiment)]
    if subset.empty:
        return subset
    return (
        subset.sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["sentence"])
        .head(k)[["sentence", "confidence", "review_idx"]]
    )


def render_synthesis(df: pd.DataFrame, top_n_priority: int = 5):
    synth = build_aspect_synthesis(df)
    if synth.empty:
        st.warning("Aucune prediction exploitable.")
        return

    n_reviews = df["review_idx"].nunique()
    n_sentences = df[["review_idx", "sentence_idx"]].drop_duplicates().shape[0]
    n_preds = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avis analyses", n_reviews)
    c2.metric("Phrases", n_sentences)
    c3.metric("Predictions (aspect x phrase)", n_preds)
    c4.metric("Aspects touches", synth.shape[0])

    st.subheader("Synthese par aspect")
    st.caption(
        "Tri par priorite = nombre de mentions negatives "
        "(axes d'amelioration potentiels)."
    )
    display = synth.rename(
        columns={
            "aspect": "Aspect",
            "mentions": "Mentions",
            "n_positive": "Positifs",
            "n_negative": "Negatifs",
            "pct_positive": "% Positif",
            "pct_negative": "% Negatif",
            "priority_score": "Priorite (nb neg.)",
            "avg_confidence": "Confiance moy.",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Graphique simple
    chart_df = synth.set_index("aspect")[["pct_positive", "pct_negative"]]
    st.bar_chart(chart_df, stack=False)

    st.subheader("Aspects a surveiller en priorite")
    priority = synth[synth["n_negative"] > 0].head(top_n_priority)
    if priority.empty:
        st.success("Aucun aspect majoritairement negatif detecte sur ce corpus.")
    else:
        for _, row in priority.iterrows():
            with st.expander(
                f"🔴 {row['aspect']} — {row['n_negative']} negatif(s) "
                f"/ {row['mentions']} mention(s) ({row['pct_negative']:.0f}% neg.)",
                expanded=False,
            ):
                st.markdown("**Verbatims negatifs representatifs**")
                neg_v = pick_verbatims(df, row["aspect"], "negative", k=3)
                if neg_v.empty:
                    st.write("_Aucun verbatim negatif._")
                else:
                    for _, v in neg_v.iterrows():
                        st.markdown(
                            f"- « {v['sentence']} » "
                            f"_(avis #{int(v['review_idx'])}, "
                            f"conf. {v['confidence']*100:.0f}%)_"
                        )

                st.markdown("**Verbatims positifs (contraste)**")
                pos_v = pick_verbatims(df, row["aspect"], "positive", k=2)
                if pos_v.empty:
                    st.write("_Aucun verbatim positif._")
                else:
                    for _, v in pos_v.iterrows():
                        st.markdown(
                            f"- « {v['sentence']} » "
                            f"_(avis #{int(v['review_idx'])}, "
                            f"conf. {v['confidence']*100:.0f}%)_"
                        )

    st.subheader("Points forts (aspects les plus positifs)")
    strengths = (
        synth[synth["mentions"] >= 2]
        .sort_values(["pct_positive", "mentions"], ascending=[False, False])
        .head(5)
    )
    if strengths.empty:
        st.write("_Pas assez de mentions pour identifier des points forts._")
    else:
        for _, row in strengths.iterrows():
            st.markdown(
                f"- 🟢 **{row['aspect']}** : {row['pct_positive']:.0f}% positif "
                f"({row['n_positive']}/{row['mentions']})"
            )

    st.download_button(
        "Telecharger la synthese CSV",
        data=synth.to_csv(index=False).encode("utf-8"),
        file_name="absa_synthesis_by_aspect.csv",
        mime="text/csv",
    )
    st.download_button(
        "Telecharger le detail des predictions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="absa_predictions_detail.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.title("Synthese d'avis clients — Aspect-Based Sentiment Analysis")
    st.caption(
        "BERT fine-tune (polarite) + detection d'aspects (cues lexicaux) "
        "pour identifier points forts et axes d'amelioration sur un corpus."
    )

    try:
        model, tokenizer, id2label, max_len = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    taxonomy = load_aspect_taxonomy()

    with st.sidebar:
        st.header("Parametres")
        st.caption(f"Device: `{DEVICE}`")
        auto_detect = st.checkbox(
            "Detecter automatiquement les aspects (recommandé)",
            value=True,
            help=(
                "Utilise les indices lexicaux du projet (meme logique que le "
                "labeling). Evite de predire un sentiment sur un aspect absent."
            ),
        )
        forced_aspects = st.multiselect(
            "Aspects forces (optionnel)",
            options=taxonomy,
            default=[],
            help="Evalues systematiquement, meme sans cue lexical.",
        )
        top_n = st.slider("Top aspects prioritaires", 3, 10, 5)
        st.info(
            "CSV attendu : une ligne = un avis, colonne "
            "`text` / `review` / `review_text` / `sentence` / `avis`."
        )

    tab_corpus, tab_single = st.tabs(
        ["Corpus (CSV / multi-avis)", "Avis unique (demo rapide)"]
    )

    # ----------------------------- CORPUS ---------------------------------
    with tab_corpus:
        st.subheader("Injecter un corpus d'avis")
        uploaded = st.file_uploader("Uploader un CSV", type=["csv"])
        pasted = st.text_area(
            "Ou coller plusieurs avis (une ligne = un avis)",
            height=180,
            placeholder=(
                "The plate is cute but the suction fails quickly.\n"
                "Great value for the price and very soft fabric.\n"
                "Shipping was late and packaging arrived damaged."
            ),
        )

        run_corpus = st.button("Analyser le corpus", type="primary", key="run_corpus")
        if run_corpus:
            reviews: list[str] = []
            if uploaded is not None:
                reviews, err = parse_uploaded_csv(uploaded)
                if err:
                    st.error(err)
                    st.stop()
            elif pasted.strip():
                reviews = parse_pasted_reviews(pasted)
            else:
                st.warning("Uploadez un CSV ou collez des avis.")
                st.stop()

            if not reviews:
                st.warning("Aucun avis valide detecte.")
                st.stop()
            if not auto_detect and not forced_aspects:
                st.warning(
                    "Activez la detection automatique ou forcez au moins un aspect."
                )
                st.stop()

            st.write(f"**{len(reviews)} avis** charges — lancement de l'analyse...")
            with st.spinner("Inference BERT en cours..."):
                detail_df = analyze_corpus(
                    reviews=reviews,
                    model=model,
                    tokenizer=tokenizer,
                    id2label=id2label,
                    max_len=max_len,
                    taxonomy=taxonomy,
                    forced_aspects=forced_aspects,
                    auto_detect=auto_detect,
                )

            if detail_df.empty:
                st.warning("Aucune paire (aspect, phrase) a predire.")
                st.stop()

            render_synthesis(detail_df, top_n_priority=top_n)

    # ----------------------------- SINGLE ---------------------------------
    with tab_single:
        st.subheader("Tester un avis isole")
        review_text = st.text_area(
            "Avis produit",
            height=140,
            key="single_review",
            placeholder=(
                "The plate is super cute but the suction doesn't hold at all "
                "after a few uses. Price is fair for the quality though."
            ),
        )
        run_single = st.button("Analyser cet avis", key="run_single")
        if run_single:
            if not review_text.strip():
                st.warning("Saisissez un avis.")
                st.stop()
            if not auto_detect and not forced_aspects:
                st.warning(
                    "Activez la detection automatique ou forcez au moins un aspect."
                )
                st.stop()

            with st.spinner("Analyse..."):
                detail_df = analyze_corpus(
                    reviews=[review_text.strip()],
                    model=model,
                    tokenizer=tokenizer,
                    id2label=id2label,
                    max_len=max_len,
                    taxonomy=taxonomy,
                    forced_aspects=forced_aspects,
                    auto_detect=auto_detect,
                )
            if detail_df.empty:
                st.warning("Aucune prediction.")
                st.stop()
            render_synthesis(detail_df, top_n_priority=top_n)

            with st.expander("Detail phrase x aspect"):
                st.dataframe(
                    detail_df[
                        ["sentence", "aspect", "sentiment", "confidence"]
                    ].rename(
                        columns={
                            "sentence": "Phrase",
                            "aspect": "Aspect",
                            "sentiment": "Sentiment",
                            "confidence": "Confiance",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
