"""
Préparation du dataset ABSA (Baby Products) — Étapes : nettoyage, tokenisation, split.

Format d'entree (JSONL, une ligne = phrase x aspect) :
{
    "review_idx": 1357, "sentence": "...", "aspect": "...",
    "aspect_score": 0.99, "sentiment": "positive", ...
}

Note : "sentence" (la phrase segmentée) est utilisé comme texte de travail,
PAS "review_text" (l'avis complet brut) — c'est sur "sentence" que portent
l'aspect et le sentiment annotés.

Exemples :
    # Smoke test sur 100 avis du fichier final
    python scripts/data_prep/prep_data.py \\
        --input Data/Baby_Products_aspect_labels_flat_final_zs_model.jsonl \\
        --max-reviews 100

    # Pipeline complet
    python scripts/data_prep/prep_data.py \\
        --input Data/Baby_Products_aspect_labels_flat_final_zs_model.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "Data" / "Baby_Products_aspect_labels_flat_final_zs_model.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_splits"

COL_TEXT = "sentence"
COL_ASPECT = "aspect"
COL_SENTIMENT = "sentiment"
COL_GROUP = "review_idx"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Etape 1 — Nettoyage
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Nettoyage minimal non destructif (casse/ponctuation conservees pour BERT)."""
    if not isinstance(text, str):
        return ""

    # 1. Décoder les entités HTML résiduelles (&#34; -> ", &amp; -> &, etc.)
    #    Très présent dans les avis Amazon bruts, vu dans votre échantillon.
    text = html.unescape(text)

    # 2. Retirer les balises HTML résiduelles (ex: <br />, <br/>) — la
    #    segmentation en phrases peut en laisser en fin/début de phrase.
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Normaliser les guillemets et apostrophes typographiques vers leurs
    #    équivalents ASCII (’ -> ', " " -> ", … -> ...). Sans ça, le
    #    tokenizer regex \w+ traite "don’t" et "don't" comme deux tokens
    #    différents alors que ce sont le même mot.
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2026", "...")

    # 4. Retirer les URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 5. Normaliser les espaces multiples / retours à la ligne / tabulations
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Réduire les répétitions excessives de caractères (ex: "greeaaaat" -> "greaat")
    #    Fréquent dans les avis très enthousiastes ("SOOOO CUTE" / "YESSS"),
    #    on garde une trace de l'emphase (2 répétitions) sans polluer le vocabulaire.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons exacts (texte + aspect + sentiment)."""
    before = len(df)
    df = df.drop_duplicates(subset=[COL_TEXT, COL_ASPECT, COL_SENTIMENT])
    df = df[df[COL_TEXT].str.len() > 0]
    print(f"[clean] {before - len(df)} lignes supprimees (doublons / texte vide)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Étape 2 — Tokenisation (pour le LSTM baseline)
# ---------------------------------------------------------------------------
# Note importante : le BERT fine-tuning (phase 3) NE PASSE PAS par cette
# tokenisation. BertTokenizer fait sa propre tokenisation sous-mots (WordPiece)
# directement sur le texte nettoyé de l'étape 1. Cette section ne sert
# qu'à préparer le vocabulaire du LSTM.

def simple_tokenize(text: str) -> list[str]:
    """
    Tokenisation légère par regex, suffisante pour un LSTM baseline.
    Si vous voulez une tokenisation linguistique plus fine (lemmatisation,
    gestion des contractions "don't" -> "do", "n't"), remplacez cette fonction
    par un appel à spaCy : [t.text for t in nlp(text.lower())]
    """
    text = text.lower()
    # Inclut l'apostrophe interne pour garder "don't", "it's" comme un seul token
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def build_vocab(
    token_lists: list[list[str]], min_freq: int = 2, max_size: int = 20000
) -> dict[str, int]:
    """Vocabulaire construit sur le TRAIN uniquement (evite la fuite)."""
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # Tokens spéciaux obligatoires pour un LSTM avec padding
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(
        f"[vocab] {len(vocab)} tokens retenus (min_freq={min_freq}) "
        f"sur {len(counter)} tokens uniques observés"
    )
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(tok, unk) for tok in tokens]


# ---------------------------------------------------------------------------
# Etape 3 — Split groupé (review_idx) train / val / test SANS FUITE DE DONNÉES
# ---------------------------------------------------------------------------
def split_by_review_group(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split au niveau review_idx pour eviter qu'un meme avis fuite
    entre train / val / test (plusieurs phrases peuvent être issues du même avis).
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size doit valoir 1.0")

    if COL_GROUP not in df.columns:
        raise ValueError(
            f"Colonne '{COL_GROUP}' absente. Impossible de garantir "
            "l'absence de fuite entre splits."
        )

    n_groups = df[COL_GROUP].nunique()
    if n_groups < 3:
        raise ValueError(
            f"Seulement {n_groups} avis distincts (review_idx). "
            "Il en faut au moins 3 pour un split train/val/test groupe. "
            "Utilisez --max-reviews >= 30 sur le fichier final, "
            "ou un echantillon contenant plusieurs avis."
        )

    # 1) train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df[COL_GROUP]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # 2) val vs test sur le reste
    #    Si trop peu de groupes dans temp (echantillon tout petit), on reparti
    #    manuellement les groupes pour garantir 1+ / 1+.
    temp_groups = temp_df[COL_GROUP].unique()
    relative_val = val_size / (val_size + test_size)

    if len(temp_groups) < 2:
        # Cas limite : tout le hold-out est tombe dans un seul avis.
        # On re-split depuis tous les groupes non-train.
        all_groups = df[COL_GROUP].unique().tolist()
        train_groups = set(train_df[COL_GROUP])
        holdout = [g for g in all_groups if g not in train_groups]
        if len(holdout) < 2:
            # Forcer au moins 1 groupe val et 1 test en piochant dans train
            rng_groups = sorted(all_groups)
            if len(rng_groups) < 3:
                raise ValueError("Pas assez de groupes pour forcer val/test.")
            test_g = rng_groups[-1]
            val_g = rng_groups[-2]
            train_keep = set(rng_groups[:-2])
            train_df = df[df[COL_GROUP].isin(train_keep)].reset_index(drop=True)
            val_df = df[df[COL_GROUP] == val_g].reset_index(drop=True)
            test_df = df[df[COL_GROUP] == test_g].reset_index(drop=True)
        else:
            val_g, test_g = holdout[0], holdout[1]
            val_df = df[df[COL_GROUP] == val_g].reset_index(drop=True)
            test_df = df[df[COL_GROUP] == test_g].reset_index(drop=True)
    else:
        gss2 = GroupShuffleSplit(
            n_splits=1, train_size=relative_val, random_state=seed
        )
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[COL_GROUP]))
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Vérification explicite qu'aucun review_id ne fuit entre les splits
    train_groups = set(train_df[COL_GROUP])
    val_groups = set(val_df[COL_GROUP])
    test_groups = set(test_df[COL_GROUP])
    assert not (train_groups & val_groups), "Fuite train/val detectee"
    assert not (train_groups & test_groups), "Fuite train/test detectee"
    assert not (val_groups & test_groups), "Fuite val/test detectee"

    print(
        f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(groupes: {len(train_groups)}/{len(val_groups)}/{len(test_groups)})"
    )
    return train_df, val_df, test_df


def report_label_distribution(df: pd.DataFrame, name: str) -> None:
    print(f"\n[distribution] {name}")
    print(df.groupby([COL_ASPECT, COL_SENTIMENT]).size().unstack(fill_value=0))


# ---------------------------------------------------------------------------
# Pipeline principal (chargement / echantillonnage)
# ---------------------------------------------------------------------------
def load_dataset(
    input_path: Path,
    max_reviews: int | None = None,
    min_aspect_score: float | None = None,
    drop_weak: bool = False,
    drop_neutral: bool = True,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    print(f"Chargement de {input_path}")
    df = pd.read_json(input_path, lines=True)
    print(f"[load] {len(df)} lignes, {df[COL_GROUP].nunique()} avis distincts")

    required = {COL_TEXT, COL_ASPECT, COL_SENTIMENT, COL_GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans l'entree: {sorted(missing)}")

    if min_aspect_score is not None and "aspect_score" in df.columns:
        before = len(df)
        df = df[df["aspect_score"] >= min_aspect_score]
        print(
            f"[filter] {before - len(df)} lignes retirees "
            f"(aspect_score < {min_aspect_score})"
        )

    if drop_weak and "aspect_weak" in df.columns:
        before = len(df)
        df = df[~df["aspect_weak"].astype(bool)]
        print(f"[filter] {before - len(df)} lignes retirees (aspect_weak=True)")

    # Classification binaire : on ne garde que positive / negative
    if drop_neutral:
        before = len(df)
        df = df[df[COL_SENTIMENT].isin(["positive", "negative"])]
        print(
            f"[filter] {before - len(df)} lignes retirees "
            f"(sentiment hors {{positive, negative}})"
        )

    if max_reviews is not None:
        groups = (
            df[COL_GROUP]
            .drop_duplicates()
            .sample(n=min(max_reviews, df[COL_GROUP].nunique()), random_state=seed)
        )
        before = len(df)
        df = df[df[COL_GROUP].isin(groups)]
        print(
            f"[sample] {df[COL_GROUP].nunique()} avis gardes "
            f"({len(df)}/{before} lignes)"
        )

    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preparation ABSA: nettoyage, tokenisation LSTM, split groupe"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Limite le nombre d'avis (review_idx) pour un smoke test",
    )
    p.add_argument(
        "--min-aspect-score",
        type=float,
        default=None,
        help="Filtre optionnel, ex. 0.7",
    )
    p.add_argument(
        "--drop-weak",
        action="store_true",
        help="Retire les lignes avec aspect_weak=True",
    )
    p.add_argument(
        "--keep-neutral",
        action="store_true",
        help="Conserve les lignes 'neutral' (desactive par defaut le filtre binaire)",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(
        input_path=args.input,
        max_reviews=args.max_reviews,
        min_aspect_score=args.min_aspect_score,
        drop_weak=args.drop_weak,
        drop_neutral=not args.keep_neutral,
        seed=args.seed,
    )

    # 1. Nettoyage
    df[COL_TEXT] = df[COL_TEXT].apply(clean_text)
    df = deduplicate(df)

    # 2. Split AVANT le vocabulaire (evite la fuite)
    train_df, val_df, test_df = split_by_review_group(df, seed=args.seed)

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        report_label_distribution(split_df, split_name)

    # 3. Tokenisation + vocabulaire (TRAIN uniquement)
    train_tokens = train_df[COL_TEXT].apply(simple_tokenize)
    vocab = build_vocab(
        train_tokens.tolist(), min_freq=args.min_freq, max_size=args.max_vocab
    )

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out = split_df.copy()
        out["tokens"] = out[COL_TEXT].apply(simple_tokenize)
        out["token_ids"] = out["tokens"].apply(lambda t: encode_tokens(t, vocab))
        out_path = args.output_dir / f"{split_name}.jsonl"
        out.to_json(out_path, orient="records", lines=True, force_ascii=False)
        print(f"[write] {out_path} ({len(out)} lignes)")

    vocab_path = args.output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[write] {vocab_path}")
    print(f"\nTermine. Fichiers dans {args.output_dir}/")


if __name__ == "__main__":
    main()
