"""
Phase 3 — Fine-tuning BERT pour ABSA (classification binaire de sentiment par aspect).

Difference cle avec le baseline Bi-LSTM (train_baseline.py) :

    Baseline LSTM : la phrase et l'aspect sont encodes SEPAREMENT puis
    concatenes tardivement.

    BERT (ce script) : paire [CLS] aspect [SEP] phrase [SEP] (BERT-SPC).

Prerequis : data/splits/{train,val,test}.jsonl produits par prep_data.py
(champ texte brut "sentence", pas les token_ids LSTM).

Exemples :
    python src/train_bert.py --max-epochs 1
    python src/train_bert.py --data-dir data/splits --model-dir models/bert --output-dir outputs/bert
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "bert"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bert"

TEXT_FIELD = "sentence"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 96
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_EPOCHS = 4
PATIENCE = 2
GRAD_CLIP_NORM = 1.0

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Etape 1 — Mapping des labels (binaire : negative / positive)
# ---------------------------------------------------------------------------
def build_label_mapping(train_path: Path) -> dict[str, int]:
    sentiments = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            sentiments.add(json.loads(line)["sentiment"])

    order = [s for s in ["negative", "positive"] if s in sentiments]
    unexpected = sentiments - set(order)
    if unexpected:
        print(f"[labels] sentiments ignores (hors binaire): {sorted(unexpected)}")

    label2id = {s: i for i, s in enumerate(order)}
    if len(label2id) != 2:
        raise ValueError(
            f"Attendu 2 sentiments (negative, positive), trouve: {list(label2id)}. "
            "Relancez prep_data.py (filtre neutral active par defaut)."
        )
    print(f"[labels] {len(label2id)} classes : {list(label2id.keys())}")
    return label2id


# ---------------------------------------------------------------------------
# Etape 2 — Dataset : tokenisation par paire (aspect, phrase)
# ---------------------------------------------------------------------------
class ABSABertDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        label2id: dict[str, int],
        max_len: int = MAX_LEN,
    ):
        self.examples: list[tuple[str, str, int]] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sentiment = row["sentiment"]
                if sentiment not in label2id:
                    skipped += 1
                    continue
                text = row.get(TEXT_FIELD) or ""
                if not str(text).strip():
                    skipped += 1
                    continue
                self.examples.append((row["aspect"], text, label2id[sentiment]))

        if skipped:
            print(f"[dataset:{path.name}] {skipped} lignes ignorees")
        print(f"[dataset:{path.name}] {len(self.examples)} exemples")

        aspects = [ex[0] for ex in self.examples]
        sentences = [ex[1] for ex in self.examples]

        encodings = tokenizer(
            text=aspects,
            text_pair=sentences,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings["token_type_ids"]
        self.labels = torch.tensor([ex[2] for ex in self.examples], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Etape 3 — Boucle d'entrainement
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(batch["labels"])
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item() * len(batch["labels"])

        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


def build_optimizer(model, lr: float = LEARNING_RATE):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(grouped_params, lr=lr)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning BERT ABSA (binaire)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else REPO_ROOT / args.data_dir
    model_dir = (
        args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model_state.pt"

    required = ["train.jsonl", "val.jsonl", "test.jsonl"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers manquants dans {data_dir}: {missing}. "
            "Lancez d'abord: python src/prep_data.py"
        )

    print(f"Device utilise : {DEVICE}")
    print(f"DATA_DIR   : {data_dir}")
    print(f"MODEL_DIR  : {model_dir}")
    print(f"OUTPUT_DIR : {output_dir}")
    if DEVICE.type == "cpu":
        print("[warning] Pas de GPU : fine-tuning BERT lent sur CPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(model_dir / "tokenizer")

    label2id = build_label_mapping(data_dir / "train.jsonl")
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    train_ds = ABSABertDataset(
        data_dir / "train.jsonl", tokenizer, label2id, max_len=args.max_len
    )
    val_ds = ABSABertDataset(
        data_dir / "val.jsonl", tokenizer, label2id, max_len=args.max_len
    )
    test_ds = ABSABertDataset(
        data_dir / "test.jsonl", tokenizer, label2id, max_len=args.max_len
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Dataset vide (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    ).to(DEVICE)

    optimizer = build_optimizer(model, lr=args.lr)
    total_steps = max(1, len(train_loader) * args.max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "model_name": args.model_name,
                    "max_len": args.max_len,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"[early stopping] pas d'amelioration du F1 val depuis "
                    f"{args.patience} epochs, arret."
                )
                break

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(
        model, test_loader
    )
    print(f"\n=== Resultats test (meilleur modele, val_f1={best_val_f1:.4f}) ===")
    print(
        f"accuracy={test_acc:.4f} | macro_precision={test_prec:.4f} "
        f"| macro_recall={test_rec:.4f} | macro_f1={test_f1:.4f}"
    )

    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print(report)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": test_acc,
                "macro_precision": test_prec,
                "macro_recall": test_rec,
                "macro_f1": test_f1,
                "classification_report": report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (lignes=vrai, colonnes=predit) :")
    print(target_names)
    print(cm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss (BERT)")
    axes[0].legend()

    axes[1].plot(history["val_f1"], label="val_macro_f1", color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Macro F1 (validation) — BERT")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    print(f"\nCourbes sauvegardees dans {output_dir / 'training_curves.png'}")
    print(f"Modele sauvegarde dans {checkpoint_path}")
    print(f"Tokenizer sauvegarde dans {model_dir / 'tokenizer'}")
    print(f"Metriques sauvegardees dans {output_dir}/")


if __name__ == "__main__":
    main()

