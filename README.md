# ABSA — Analyse automatique d'avis clients (Aspect-Based Sentiment Analysis)

Projet réalisé dans le cadre du cours *Advanced Practical Machine Learning* — MSc Data Engineering & Cloud Computing, aivancity School of AI & Data.

## Le problème

Les entreprises reçoivent chaque jour un volume important d'avis clients qu'il est difficile d'analyser manuellement. Une note globale ne dit pas grand-chose des points forts et des axes d'amélioration réels d'un produit ou d'un service.

Ce projet entraîne un modèle de Deep Learning (PyTorch, fine-tuning BERT) capable de déterminer, pour un aspect donné (sécurité, prix, qualité, facilité d'usage...), le sentiment exprimé à son sujet dans un avis. Une démo Streamlit permet d'analyser un avis à la fois, ou d'injecter un batch de plusieurs dizaines d'avis pour obtenir une synthèse agrégée par aspect.

## Démarrage rapide — utiliser le modèle déjà entraîné

**Vous n'avez pas besoin de ré-entraîner quoi que ce soit pour tester le projet.** Le modèle BERT fine-tuné est fourni directement dans ce dépôt.

```bash
# 1) Clone + LFS (si pas déjà fait)
git clone https://github.com/Lionel-Karl-AIVANCITY/advanced-machine-learning-project.git
cd advanced-machine-learning-project

# Le checkpoint BERT (~400 Mo) est suivi via Git LFS — assurez-vous de l'avoir installé
git lfs install
# avant le clone (https://git-lfs.com), sinon récupérez-le après coup :
git lfs pull

# 2) Environnement
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS/Linux :
# source .venv/bin/activate

pip install -r requirements.txt

# 3) Démo
streamlit run app/app.py
```

L'application s'ouvre dans le navigateur avec deux modes :
- **Avis unique** : collez un avis, choisissez les aspects à tester
- **Analyse en masse** : importez un CSV de plusieurs avis, ou collez-en plusieurs (un par ligne) — les aspects sont détectés automatiquement et une synthèse agrégée est produite (aspects à surveiller en priorité, verbatims représentatifs, export CSV)

Un petit échantillon prêt à l'emploi pour tester le mode batch est fourni dans `data/samples/testing_demo_example.csv`.

## Structure du dépôt

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

## Reproduire le pipeline complet (optionnel)

Cette section n'est nécessaire que si vous voulez ré-entraîner les modèles vous-même — pas requis pour tester la démo.

### 1. Données

Le dataset utilisé est la catégorie **Baby_Products** du corpus [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab).

> Le lien suivant vous envoit sur la page contenant les données [Lien vers les données hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories)
> téléchargez le fichier de la catégorie "Baby_Products" et placez-le dans data/raw/


Voir `data/README.md` pour le détail et les précautions de licence (usage académique, pas de redistribution du texte brut des avis).

### 2. Préparation des données

```bash
python src/prep_data.py
```
Produit `data_splits/{train,val,test}.jsonl` + `vocab.json`.

### 3. Baseline Bi-LSTM

```bash
python src/train_baseline.py
```

### 4. Fine-tuning BERT

```bash
python src/train_bert.py
```
GPU fortement recommandé (Colab/Kaggle si pas de GPU local).

### 5. Analyse d'erreurs

```bash
python src/error_analysis.py
```

### 6. Lancement de l'application streamlit

```bash
streamlit run app/app.py
```

## Résultats

| Modèle | Accuracy | Macro F1 | F1 negative | F1 positive |
|---|---|---|---|---|
| Baseline Bi-LSTM | 0.818 | 0.803 | 0.75 | 0.86 |
| BERT fine-tuné | **0.929** | **0.922** | **0.90** | **0.95** |

Classification binaire (positive/négatif) — la classe `neutral` a été retirée après inspection manuelle ayant montré qu'elle correspondait en réalité à des labels bruités plutôt qu'à une vraie catégorie de sentiment (détails dans le rapport).

## Limites connues

- Le modèle est entraîné sur du vocabulaire e-commerce (Baby Products) et ne généralise pas nécessairement à d'autres domaines (restauration, hôtellerie) sans fine-tuning additionnel.
- La détection automatique d'aspect (mode "Analyse en masse") s'appuie sur un classifieur zero-shot générique (`facebook/bart-large-mnli`), plus lent et moins précis qu'un modèle spécifiquement entraîné pour cette tâche.
- Le modèle prédit un sentiment pour tout aspect qu'on lui soumet, même s'il n'est pas réellement mentionné dans la phrase (mode "Avis unique" uniquement — le mode batch filtre via la détection d'aspect).

## Documentation

Voir `docs/` (cadrage, stack technique, notebook de comparaison des modèles).


## Équipe

Lionel Jospin KENNE NZANGEM & Karl Sondeji — MSc 2 Data Engineering & Cloud Computing, aivancity School of AI & Data (2025–2026). Tuteur : Souheil Hanoune.
