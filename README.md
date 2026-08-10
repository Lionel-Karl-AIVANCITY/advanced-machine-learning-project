# ABSA — Analyse d'avis clients (Baby Products)

Projet *Advanced Practical Machine Learning* (aivancity) :  
**Aspect-Based Sentiment Analysis** sur des avis Amazon, avec baseline Bi-LSTM, fine-tuning BERT et démo Streamlit.

## Structure

```
Machine-learning-project-lionel-karl/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── prep_data.py
│   ├── train_baseline.py
│   ├── train_bert.py
│   ├── error_analysis.py
│   ├── build_aspect_labels.py
│   └── labeling/                 # pipeline zero-shot (interne)
├── app/
│   └── app.py
├── data/
│   ├── README.md
│   ├── samples/testing_demo_example.csv
│   ├── raw/                      # gitignored
│   └── splits/                   # gitignored
├── models/
│   ├── baseline/best_model.pt
│   └── bert/                     # poids lourds gitignored
├── outputs/
│   ├── baseline/
│   ├── bert/
│   └── error_analysis/
├── docs/
└── report/
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline

```bash
# 1) Labels aspect-level (si besoin)
python src/build_aspect_labels.py

# 2) Prep + split
python src/prep_data.py

# 3) Baseline Bi-LSTM
python src/train_baseline.py

# 4) BERT
python src/train_bert.py

# 5) Analyse d'erreurs
python src/error_analysis.py

# 6) Démo
streamlit run app/app.py
```

## Démo Streamlit

Uploader un CSV (`review` / `text` / …) ou coller plusieurs avis pour obtenir une **synthèse par aspect** (taux pos/nég, priorités, verbatims).  
Échantillon fourni : `data/samples/testing_demo_example.csv`.

## Documentation

Voir `docs/` (cadrage, stack technique, notebook de comparaison des modèles).
