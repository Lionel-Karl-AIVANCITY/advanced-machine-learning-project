# Stack technique — Analyse automatique d'avis clients (ABSA)

Détail des outils, bibliothèques et frameworks nécessaires pour les phases 1 à 4 du projet.

---

## Environnement général (transverse aux 4 phases)

| Outil | Rôle |
|---|---|
| **Python 3.10+** | Langage principal |
| **conda** ou **venv** + `pip` | Isolation de l'environnement |
| **Git / GitHub** | Versioning, livrable final du repo |
| **Jupyter Notebook / JupyterLab** | Exploration et prototypage |
| **VS Code** (ou équivalent) | Développement des scripts finaux |
| **CUDA / GPU (Colab, Kaggle Notebooks ou GPU local)** | Indispensable pour le fine-tuning BERT (phase 3) — le CPU seul sera trop lent |

---

## Phase 1 — Données (collecte, nettoyage, préparation)

**Objectif** : obtenir un dataset avec annotations par aspect, le nettoyer et le préparer pour l'entraînement.

| Outil | Rôle |
|---|---|
| `datasets` (Hugging Face) | Accès direct aux datasets Amazon Reviews 2023|
| `pandas` | Manipulation tabulaire, dédoublonnage, exploration |
| `requests` / `beautifulsoup4` | Si scraping complémentaire nécessaire (ex. Yelp, Amazon Reviews hors dataset officiel) |
| `re` (stdlib) | Nettoyage regex (caractères spéciaux, HTML résiduel) |
| `spaCy` ou `nltk` | Tokenisation linguistique, lemmatisation, détection de langue |
| `scikit-learn` (`train_test_split`) | Split train/val/test — **usage autorisé en préparation de données, pas comme modèle principal** |
| `unicodedata` (stdlib) | Normalisation des caractères (accents, encodages) |
| `matplotlib` / `seaborn` | Visualisation exploratoire (distribution des aspects, longueur des avis, classes déséquilibrées) |

**Point de vigilance technique** : si le dataset choisi (Amazon Reviews) n'a pas nativement de labels par aspect au format exploitable, il faudra un format d'annotation intermédiaire — typiquement JSON avec structure `{"text": ..., "aspects": [{"term": ..., "sentiment": ...}]}`.

---

## Phase 2 — Baseline (LSTM / Bi-LSTM)

**Objectif** : implémenter et entraîner un modèle de référence en PyTorch pur.

| Outil | Rôle |
|---|---|
| `torch` (PyTorch) | Framework principal — `nn.LSTM`, `nn.Embedding`, boucle d'entraînement manuelle |
| `torchtext` | Vocabulaire, padding, `DataLoader` pour séquences texte (ou `torch.utils.data.Dataset` custom si `torchtext` trop contraignant) |
| `gensim` (optionnel) | Embeddings pré-entraînés (Word2Vec, GloVe) comme couche d'entrée du Bi-LSTM |
| `numpy` | Manipulation de matrices d'embeddings |
| `scikit-learn` (`metrics`) | Calcul Accuracy, Precision, Recall, F1-score — **uniquement pour l'évaluation, pas comme modèle** |
| `matplotlib` | Courbes de loss/accuracy pendant l'entraînement |
| `tqdm` | Barres de progression pour la boucle d'entraînement |

**Point de vigilance technique** : la boucle d'entraînement (forward, loss, backward, optimizer.step()) doit être écrite et comprise à la main — c'est une exigence explicite du règlement, pas seulement de la performance.

---

## Phase 3 — Modèle avancé (fine-tuning BERT)

**Objectif** : fine-tuner un Transformer pré-entraîné pour la tâche ABSA.

| Outil | Rôle |
|---|---|
| `transformers` (Hugging Face) | `BertModel`, `BertTokenizer`, chargement du modèle pré-entraîné (`bert-base-uncased` ou variante multilingue si avis en français) |
| `torch` | Boucle de fine-tuning, `AdamW`, gestion du learning rate scheduler |
| `datasets` (Hugging Face) | Formatage efficace pour l'entraînement par batch |
| `accelerate` (optionnel) | Simplifie l'utilisation GPU/mixed precision si besoin de vitesse |
| `scikit-learn` (`metrics`) | Comparaison chiffrée baseline vs BERT |
| `wandb` ou `tensorboard` | Suivi des runs d'entraînement, comparaison des expériences — utile pour l'analyse critique du rapport |

**Point de vigilance technique** : le fine-tuning pour ABSA n'est pas un fine-tuning de classification standard — il faut soit une tête de sortie multi-tâches (extraction d'aspect + classification de sentiment), soit une reformulation en classification de paires (aspect, phrase). C'est le point le plus risqué du projet techniquement — à prototyper tôt sur un petit sous-échantillon.

---

## Phase 4 — Démo (Streamlit)

**Objectif** : application interactive où un utilisateur saisit un avis et visualise les aspects détectés + sentiments associés.

| Outil | Rôle |
|---|---|
| `streamlit` | Framework de l'application web interactive |
| `torch` | Chargement du modèle entraîné (`.pt` / `.pth`) pour l'inférence |
| `transformers` | Rechargement du tokenizer et du modèle BERT fine-tuné |
| `plotly` ou `altair` | Visualisation des résultats (barres de sentiment par aspect, plus interactif que matplotlib dans Streamlit) |
| `pickle` / `torch.save` | Sérialisation du modèle et du vocabulaire/tokenizer |

**Point de vigilance technique** : prévoir un format d'inférence unique et testé tôt (phase 2bis "démo précoce" sur la baseline) pour éviter de découvrir des incompatibilités d'input/output au moment de brancher BERT en fin de projet.

---

## Récapitulatif — fichier `requirements.txt` indicatif

```
torch>=2.0
transformers>=4.30
datasets
torchtext
scikit-learn
pandas
numpy
spacy
nltk
gensim
matplotlib
seaborn
plotly
streamlit
tqdm
wandb
```

