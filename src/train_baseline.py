"""
Phase 2 — Baseline Bi-LSTM pour ABSA (classification de sentiment par aspect).

Formulation de la tâche :
    Entrée  : une phrase (tokens) + un aspect ciblé (ex: "safety")
    Sortie  : le sentiment associé à CET aspect dans CETTE phrase (positive/negative)

Ce n'est pas de la classification de sentiment globale : l'aspect fait partie
de l'entrée du modèle. Sans lui, le modèle ne pourrait pas distinguer
"the price is great but the safety strap is flimsy" -> price=positive, safety=negative.

Architecture (à justifier en vidéo) :
    - Embedding de mots (appris depuis zéro, vocabulaire construit sur train uniquement)
    - Embedding d'aspect (appris depuis zéro, un vecteur par catégorie d'aspect)
    - Bi-LSTM sur la séquence de mots -> capture le contexte gauche/droite de chaque mot
    - Concaténation [dernier état caché Bi-LSTM ; embedding d'aspect]
    - Couche linéaire (au moins une couche cachée) -> 2 classes de sentiment

Prérequis : avoir exécuté prep_data.py, qui produit data/splits/{train,val,test}.jsonl
et data/splits/vocab.json.

Exemples :
    python src/train_baseline.py
    python src/train_baseline.py --data-dir data/splits --model-dir models/baseline --output-dir outputs/baseline
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "baseline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "baseline"

EMBED_DIM = 128
ASPECT_EMBED_DIM = 32
HIDDEN_DIM = 128        # par direction ; sortie Bi-LSTM = 2 * HIDDEN_DIM
NUM_LSTM_LAYERS = 1
DROPOUT = 0.3

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5             # early stopping : arrêt si le F1 val ne progresse plus

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Étape 1 — Mappings aspect / sentiment (construits sur le train uniquement,
# même logique que le vocabulaire de mots dans prep_data.py)
# ---------------------------------------------------------------------------
def build_label_mappings(train_path: Path):
    aspects, sentiments = set(), set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            aspects.add(row["aspect"])
            sentiments.add(row["sentiment"])

    aspect2id = {a: i for i, a in enumerate(sorted(aspects))}
    # Ordre fixe : classification binaire positive / negative
    sentiment_order = [s for s in ["negative", "positive"] if s in sentiments]
    unexpected = sentiments - set(sentiment_order)
    if unexpected:
        print(f"[labels] sentiments ignores (hors binaire): {sorted(unexpected)}")
    label2id = {s: i for i, s in enumerate(sentiment_order)}
    if len(label2id) != 2:
        raise ValueError(
            f"Attendu 2 sentiments (negative, positive), trouve: {list(label2id)}. "
            "Relancez prep_data.py sans --keep-neutral pour filtrer 'neutral'."
        )

    print(f"[labels] {len(aspect2id)} aspects : {list(aspect2id.keys())}")
    print(f"[labels] {len(label2id)} sentiments : {list(label2id.keys())}")
    return aspect2id, label2id


# ---------------------------------------------------------------------------
# Étape 2 — Dataset PyTorch
# ---------------------------------------------------------------------------
class ABSADataset(Dataset):
    def __init__(self, path: Path, aspect2id: dict, label2id: dict):
        self.rows = []
        unknown_aspects = 0
        unknown_labels = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["aspect"] not in aspect2id:
                    # Un aspect présent dans val/test mais jamais vu au train :
                    # ne peut pas être appris, on l'exclut plutôt que de crasher.
                    unknown_aspects += 1
                    continue
                if row["sentiment"] not in label2id:
                    unknown_labels += 1
                    continue
                if not row.get("token_ids"):
                    continue
                self.rows.append({
                    "token_ids": row["token_ids"],
                    "aspect_id": aspect2id[row["aspect"]],
                    "label_id": label2id[row["sentiment"]],
                })
        if unknown_aspects:
            print(f"[dataset:{path.name}] {unknown_aspects} lignes ignorees (aspect inconnu du train)")
        if unknown_labels:
            print(f"[dataset:{path.name}] {unknown_labels} lignes ignorees (sentiment inconnu du train)")
        print(f"[dataset:{path.name}] {len(self.rows)} exemples charges")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(r["token_ids"], dtype=torch.long),
            torch.tensor(r["aspect_id"], dtype=torch.long),
            torch.tensor(r["label_id"], dtype=torch.long),
        )


def collate_fn(batch):
    """
    Les phrases ont des longueurs différentes : on les pad à la longueur max
    du batch (pas à une longueur fixe globale, pour ne pas gaspiller de calcul).
    On garde aussi les longueurs réelles pour pack_padded_sequence, qui évite
    au LSTM de "voir" les tokens de padding comme du vrai contenu.
    """
    token_seqs, aspect_ids, label_ids = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.long)

    # pack_padded_sequence exige un tri par longueur décroissante
    lengths, sort_idx = lengths.sort(descending=True)
    token_seqs = [token_seqs[i] for i in sort_idx]
    aspect_ids = torch.stack([aspect_ids[i] for i in sort_idx])
    label_ids = torch.stack([label_ids[i] for i in sort_idx])

    padded = pad_sequence(token_seqs, batch_first=True, padding_value=0)  # 0 = <pad> (voir prep_data.py)
    return padded, lengths, aspect_ids, label_ids


# ---------------------------------------------------------------------------
# Étape 3 — Modèle
# ---------------------------------------------------------------------------
class BiLSTMABSA(nn.Module):
    def __init__(self, vocab_size, num_aspects, num_classes,
                 embed_dim=EMBED_DIM, aspect_embed_dim=ASPECT_EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.aspect_embedding = nn.Embedding(num_aspects, aspect_embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        # Au moins une couche cachee (exigence cours) avant la tete de classification
        clf_in = hidden_dim * 2 + aspect_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids, lengths, aspect_ids):
        embedded = self.word_embedding(token_ids)          # (batch, seq_len, embed_dim)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        _, (h_n, _) = self.lstm(packed)
        # h_n : (num_layers * 2, batch, hidden_dim) -> on garde la dernière couche,
        # les deux directions : h_n[-2] = forward, h_n[-1] = backward
        sentence_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)    # (batch, hidden_dim*2)

        aspect_repr = self.aspect_embedding(aspect_ids)          # (batch, aspect_embed_dim)

        combined = torch.cat([sentence_repr, aspect_repr], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)                        # (batch, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Étape 4 — Boucle d'entraînement (écrite à la main, cf. exigence du cours)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for token_ids, lengths, aspect_ids, labels in loader:
        token_ids, aspect_ids, labels = token_ids.to(DEVICE), aspect_ids.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(token_ids, lengths, aspect_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for token_ids, lengths, aspect_ids, labels in loader:
        token_ids, aspect_ids, labels = token_ids.to(DEVICE), aspect_ids.to(DEVICE), labels.to(DEVICE)

        logits = model(token_ids, lengths, aspect_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    # macro F1 : chaque classe pese pareil (utile si negative << positive).
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entrainement baseline Bi-LSTM ABSA")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dossier contenant train/val/test.jsonl et vocab.json",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Dossier du checkpoint modele (best_model.pt)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier des metriques et courbes",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
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
    checkpoint_path = model_dir / "best_model.pt"

    required = ["train.jsonl", "val.jsonl", "test.jsonl", "vocab.json"]
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

    with open(data_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    aspect2id, label2id = build_label_mappings(data_dir / "train.jsonl")
    id2label = {v: k for k, v in label2id.items()}

    train_ds = ABSADataset(data_dir / "train.jsonl", aspect2id, label2id)
    val_ds = ABSADataset(data_dir / "val.jsonl", aspect2id, label2id)
    test_ds = ABSADataset(data_dir / "test.jsonl", aspect2id, label2id)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Dataset vide apres chargement "
            f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = BiLSTMABSA(
        vocab_size=vocab_size,
        num_aspects=len(aspect2id),
        num_classes=len(label2id),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion
        )

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
                    "vocab_size": vocab_size,
                    "aspect2id": aspect2id,
                    "label2id": label2id,
                    "hyperparams": {
                        "embed_dim": EMBED_DIM,
                        "aspect_embed_dim": ASPECT_EMBED_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "num_layers": NUM_LSTM_LAYERS,
                        "dropout": DROPOUT,
                    },
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

    # -------------------------------------------------------------------
    # Evaluation finale sur le test set, avec le MEILLEUR modele
    # -------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion
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
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["val_f1"], label="val_macro_f1", color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Macro F1 (validation)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    print(f"\nCourbes sauvegardees dans {output_dir / 'training_curves.png'}")
    print(f"Modele sauvegarde dans {checkpoint_path}")
    print(f"Metriques sauvegardees dans {output_dir}/")


if __name__ == "__main__":
    main()

