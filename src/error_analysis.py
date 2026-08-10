"""
Analyse qualitative des erreurs — comparaison baseline Bi-LSTM vs BERT.

Sorties dans error_analysis/ (ou --output-dir) :
    - categorized_errors.csv
    - samples_<categorie>.csv
    - error_rate_by_aspect.csv, error_rate_by_length.csv, negation_analysis.csv

Exemples :
    python src/error_analysis.py
    python src/error_analysis.py \\
        --data-dir data/splits \\
        --baseline-dir models/baseline \\
        --bert-dir models/bert \\
        --output-dir outputs/error_analysis
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from train_baseline import BiLSTMABSA  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_BASELINE_DIR = REPO_ROOT / "models" / "baseline"
DEFAULT_BERT_DIR = REPO_ROOT / "models" / "bert"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "error_analysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)

SAMPLE_SIZE_PER_CATEGORY = 20

NEGATION_MARKERS = [
    "not",
    "n't",
    "no ",
    "never",
    "however",
    "but ",
    "although",
    "except",
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------------------
# Chargement des modeles
# ---------------------------------------------------------------------------
def load_baseline(baseline_dir: Path):
    ckpt_path = baseline_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint baseline introuvable: {ckpt_path}. "
            "Entrainement requis (train_baseline.py)."
        )
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp = checkpoint["hyperparams"]
    model = BiLSTMABSA(
        vocab_size=checkpoint["vocab_size"],
        num_aspects=len(checkpoint["aspect2id"]),
        num_classes=len(checkpoint["label2id"]),
        embed_dim=hp["embed_dim"],
        aspect_embed_dim=hp["aspect_embed_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["aspect2id"], checkpoint["label2id"]


def load_bert(bert_dir: Path):
    ckpt_path = bert_dir / "best_model_state.pt"
    tok_dir = bert_dir / "tokenizer"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint BERT introuvable: {ckpt_path}. "
            "Entrainement requis (train_bert.py)."
        )
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer BERT introuvable: {tok_dir}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint["model_name"], num_labels=len(checkpoint["label2id"])
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint["label2id"], checkpoint["max_len"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_baseline(rows, model, aspect2id, label2id):
    id2label = {v: k for k, v in label2id.items()}
    preds = []
    for row in rows:
        if row["aspect"] not in aspect2id:
            preds.append(None)
            continue
        token_ids_list = row.get("token_ids") or []
        if not token_ids_list:
            preds.append(None)
            continue
        token_ids = torch.tensor([token_ids_list], dtype=torch.long).to(DEVICE)
        lengths = torch.tensor([len(token_ids_list)], dtype=torch.long)
        aspect_id = torch.tensor([aspect2id[row["aspect"]]], dtype=torch.long).to(
            DEVICE
        )

        logits = model(token_ids, lengths, aspect_id)
        pred_id = logits.argmax(dim=1).item()
        preds.append(id2label[pred_id])
    return preds


@torch.no_grad()
def predict_bert(rows, model, tokenizer, label2id, max_len, batch_size=32):
    id2label = {v: k for k, v in label2id.items()}
    preds = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        aspects = [r["aspect"] for r in batch]
        sentences = [r["sentence"] for r in batch]

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
        batch_preds = logits.argmax(dim=1).cpu().tolist()
        preds.extend(id2label[p] for p in batch_preds)
    return preds


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------
def categorize(true_label, baseline_pred, bert_pred):
    baseline_correct = baseline_pred == true_label
    bert_correct = bert_pred == true_label

    if baseline_correct and bert_correct:
        return "both_correct"
    if not baseline_correct and not bert_correct:
        return "both_wrong"
    if not baseline_correct and bert_correct:
        return "bert_fixes_baseline_error"
    return "bert_introduces_new_error"


def has_negation_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def length_bucket(text: str) -> str:
    n_words = len(text.split())
    if n_words <= 8:
        return "court (<=8 mots)"
    if n_words <= 20:
        return "moyen (9-20 mots)"
    return "long (>20 mots)"


def parse_args():
    p = argparse.ArgumentParser(description="Analyse d'erreurs Bi-LSTM vs BERT")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--bert-dir", type=Path, default=DEFAULT_BERT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-rows", type=int, default=None, help="Limite pour smoke rapide")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    baseline_dir = _resolve(args.baseline_dir)
    bert_dir = _resolve(args.bert_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {test_path}")

    print(f"Device utilise : {DEVICE}")
    print(f"DATA_DIR     : {data_dir}")
    print(f"BASELINE_DIR : {baseline_dir}")
    print(f"BERT_DIR     : {bert_dir}")
    print(f"OUTPUT_DIR   : {output_dir}")

    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.max_rows is not None and len(rows) >= args.max_rows:
                break
    print(f"[data] {len(rows)} exemples de test charges")

    print("[baseline] chargement du modele...")
    baseline_model, aspect2id, baseline_label2id = load_baseline(baseline_dir)
    print("[baseline] prediction...")
    baseline_preds = predict_baseline(
        rows, baseline_model, aspect2id, baseline_label2id
    )

    print("[bert] chargement du modele...")
    bert_model, tokenizer, bert_label2id, max_len = load_bert(bert_dir)
    print("[bert] prediction...")
    bert_preds = predict_bert(rows, bert_model, tokenizer, bert_label2id, max_len)

    records = []
    skipped = 0
    for row, b_pred, bert_pred in zip(rows, baseline_preds, bert_preds):
        if b_pred is None:
            skipped += 1
            continue
        # Ignore les labels hors mapping binaire si presents dans d'anciens splits
        if row["sentiment"] not in {"positive", "negative"}:
            if row["sentiment"] not in baseline_label2id and row["sentiment"] not in bert_label2id:
                skipped += 1
                continue
        records.append(
            {
                "review_idx": row.get("review_idx"),
                "sentence_idx": row.get("sentence_idx"),
                "aspect": row["aspect"],
                "sentence": row["sentence"],
                "true_label": row["sentiment"],
                "baseline_pred": b_pred,
                "bert_pred": bert_pred,
                "category": categorize(row["sentiment"], b_pred, bert_pred),
                "n_words": len(row["sentence"].split()),
                "has_negation_marker": has_negation_marker(row["sentence"]),
            }
        )

    if skipped:
        print(f"[data] {skipped} lignes ignorees (aspect/token_ids/label)")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Aucune ligne exploitable pour l'analyse d'erreurs.")

    df.to_csv(output_dir / "categorized_errors.csv", index=False)

    print("\n=== Repartition des categories ===")
    counts = df["category"].value_counts()
    for cat, n in counts.items():
        print(f"{cat:30s} {n:5d}  ({100 * n / len(df):.1f}%)")

    for cat in [
        "bert_fixes_baseline_error",
        "both_wrong",
        "bert_introduces_new_error",
    ]:
        subset = df[df["category"] == cat]
        if len(subset) == 0:
            continue
        sample = subset.sample(
            n=min(SAMPLE_SIZE_PER_CATEGORY, len(subset)), random_state=SEED
        )
        sample.to_csv(output_dir / f"samples_{cat}.csv", index=False)
        print(f"\n[echantillon] {len(sample)} exemples de '{cat}' -> samples_{cat}.csv")

    print("\n=== Taux d'erreur par aspect (BERT) ===")
    df["bert_wrong"] = df["bert_pred"] != df["true_label"]
    by_aspect = (
        df.groupby("aspect")["bert_wrong"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    by_aspect.columns = ["taux_erreur_bert", "n_exemples"]
    print(by_aspect)
    by_aspect.to_csv(output_dir / "error_rate_by_aspect.csv")

    print("\n=== Taux d'erreur par longueur de phrase (BERT) ===")
    df["length_bucket"] = df["sentence"].apply(length_bucket)
    by_length = df.groupby("length_bucket")["bert_wrong"].agg(["mean", "count"])
    by_length.columns = ["taux_erreur_bert", "n_exemples"]
    print(by_length)
    by_length.to_csv(output_dir / "error_rate_by_length.csv")

    print(
        "\n=== Taux d'erreur selon marqueur de negation/contraste (BERT) ==="
    )
    by_negation = df.groupby("has_negation_marker")["bert_wrong"].agg(
        ["mean", "count"]
    )
    by_negation.columns = ["taux_erreur_bert", "n_exemples"]
    print(by_negation)
    by_negation.to_csv(output_dir / "negation_analysis.csv")

    print(f"\nTous les fichiers sont dans {output_dir}/")


if __name__ == "__main__":
    main()
