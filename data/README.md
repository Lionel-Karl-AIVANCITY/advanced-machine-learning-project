# Données

Ce dossier contient les données du projet ABSA (avis Amazon *Baby Products*).

## Structure

| Chemin | Contenu | Versionné ? |
|--------|---------|-------------|
| `raw/` | Avis bruts + labels aspect-level (JSONL) | **Non** (gitignore) |
| `splits/` | `train.jsonl`, `val.jsonl`, `test.jsonl`, `vocab.json` | **Non** (gitignore) |
| `samples/` | Petit CSV pour tester la démo Streamlit | **Oui** |

## Obtenir / régénérer les données

1. **Avis bruts** — sous-échantillon Amazon Reviews (catégorie Baby Products), ex.  
   `data/raw/Baby_Products_echanillon.jsonl`

2. **Labels aspect-level** (zero-shot NLI + SST-2) :

```bash
python src/build_aspect_labels.py \
  --input data/raw/Baby_Products_echanillon.jsonl \
  --output-flat data/raw/Baby_Products_aspect_labels_flat_final_zs_model.jsonl
```

3. **Splits train/val/test** :

```bash
python src/prep_data.py \
  --input data/raw/Baby_Products_aspect_labels_flat_final_zs_model.jsonl \
  --output-dir data/splits
```

## Licence / usage

Les avis Amazon sont soumis aux conditions d’utilisation de la source d’origine.  
Ne pas redistribuer le corpus brut dans le dépôt GitHub public.
