# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Research Project**: "What vector representations reveal about publicistic writing: learning from mistakes" for Dialogue 2026 conference

**Goal**: Genre classification of journalistic texts from The Guardian with focus on analyzing classification errors to understand genre boundaries and gradient relationships.

**Key Hypothesis**: Classification errors reflect real genre proximity and gradient boundaries in publicistic discourse, not just model failures.

## Tech Stack

- **Python 3.9+** with PyTorch, Transformers, scikit-learn
- **NLP**: spaCy (en_core_web_sm), HuggingFace BERT/RoBERTa
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Project Structure

```
dialogue_2026/
├── data/                           # Raw and processed datasets
│   ├── cleaned_combined_guardian.csv  # 50K cleaned texts, 5 genres
│   └── combined_guardian.csv          # Raw data from Guardian API
├── notebooks/                      # Current reorganized notebooks (Phase 1)
│   ├── 01_tfidf_baseline.ipynb        # TF-IDF + Logistic Regression (86.58%)
│   ├── 02_linguistic_features.py      # Linguistic + Random Forest (65.00%)
│   ├── 03_roberta_finetuning.ipynb    # RoBERTa fine-tuning (in progress)
│   ├── run_tfidf_baseline.py          # TF-IDF runner script
│   └── run_roberta_training.py        # RoBERTa training script
├── models/                         # Trained models (gitignored)
│   ├── tfidf_lr.pkl                   # TF-IDF + LogisticRegression
│   ├── tfidf_vectorizer.pkl           # Fitted TfidfVectorizer
│   └── linguistic_rf.pkl              # Linguistic features + RandomForest
├── results/                        # Model outputs and metrics
│   ├── tfidf_metrics.json             # TF-IDF results
│   ├── linguistic_metrics.json        # Linguistic results
│   └── *.png, *.npy                   # Confusion matrices, visualizations
├── alina/                          # Legacy notebooks (original Russian author)
│   ├── cleaningTheGuard (1).ipynb     # Data cleaning pipeline
│   ├── Obuchenie (2).ipynb            # BERT training (87.62%)
│   └── WORKing (3)-Copy1.ipynb        # Guardian API data collection
├── articles/                       # Research papers and references
├── STATUS.md                       # Progress tracking (Russian)
├── plan.md                         # Detailed implementation plan (8-12 weeks)
└── classify_texts.py               # Main entry point for classification
```

## Common Development Commands

### Running Models

```bash
# Train TF-IDF baseline
python notebooks/run_tfidf_baseline.py

# Train RoBERTa model
python notebooks/run_roberta_training.py

# Classify new texts with trained models
python classify_texts.py

# Monitor RoBERTa training progress
./monitor_roberta.sh
```

### Dependencies

```bash
# Install dependencies
pip install spacy transformers torch datasets scikit-learn
pip install matplotlib seaborn statsmodels scipy jupyter

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Notebooks

```bash
# Start Jupyter
jupyter notebook

# Notebooks are in /notebooks/ directory
# Legacy notebooks in /alina/ directory (Russian, for reference)
```

## Research Architecture

### Three Levels of Linguistic Representation

1. **Lexical (TF-IDF)** - Captures word frequency patterns
   - Accuracy: 86.58%
   - Entry: `notebooks/01_tfidf_baseline.ipynb`

2. **Discourse-grammatical (Linguistic features)** - Captures syntactic and discourse patterns
   - Accuracy: 65.00%
   - Entry: `notebooks/02_linguistic_features.py`
   - Features: TTR, modal verbs, pronouns, stance markers, hedges, reporting verbs

3. **Contextual semantic (BERT/RoBERTa)** - Deep contextual representations
   - BERT: 87.62% (legacy in `alina/Obuchenie (2).ipynb`)
   - RoBERTa: ~89% expected (in progress)

### Research Phases (plan.md)

**Phase 1: Baseline Models** (Weeks 1-3)
- TF-IDF + Logistic Regression ✅
- Linguistic features + Random Forest ✅
- RoBERTa fine-tuning 🚧 (in progress)
- Comparison table (pending)

**Phase 2: Error Analysis** (Weeks 4-6)
- Confusion matrices for all models
- Identify key genre confusion pairs (e.g., News↔Analysis, Editorial↔Review)
- Extract error examples
- Qualitative linguistic analysis

**Phase 3: BERT Interpretation** (Weeks 7-8)
- Attention visualization
- Compare attention patterns across genres
- Optional: SHAP/LIME analysis

**Phase 4: Statistical Validation & Paper** (Weeks 9-12)
- McNemar's test for model comparison
- Bootstrap confidence intervals
- Inter-annotator agreement (Cohen's Kappa)
- Paper draft for Dialogue conference

## Key Files

### Documentation
- `STATUS.md` - Current progress tracking (Russian, updated regularly)
- `plan.md` - Detailed 8-12 week implementation plan
- `info.md` - Original research plan (Russian, translated)
- `NOTEBOOKS.md` - Jupyter notebook documentation
- `mistake_analysis.md` - Error analysis methodology

### Entry Points
- `classify_texts.py` - Apply trained models to new texts
- `notebooks/run_tfidf_baseline.py` - Train TF-IDF model
- `notebooks/run_roberta_training.py` - Train RoBERTa model

### Data
- `data/cleaned_combined_guardian.csv` - Main dataset (50K texts, 5 genres)
- Genres: News, Opinion, Analysis, Feature, Review

### Models
- `models/tfidf_lr.pkl` - Trained TF-IDF classifier
- `models/tfidf_vectorizer.pkl` - Fitted vectorizer
- `models/linguistic_rf.pkl` - Trained Random Forest (147 MB, gitignored)

## Important Notes

### Model Performance Hierarchy (Expected)
RoBERTa (~89%) > BERT (87.6%) > TF-IDF (86.6%) > Linguistic (65%)

### Key Research Insight
The project's unique contribution is NOT achieving maximum accuracy, but analyzing **errors** to understand:
- Which genres are most confused by models
- What linguistic features cause confusion
- How genre boundaries are gradient rather than discrete

### Data Issues
- Non-standard category formats in raw data ("News,News", "Arts,Review")
- HTML, URLs, special characters already cleaned
- Inter-annotator agreement not yet established (planned for weeks 9-10)

### GPU Requirements
- BERT/RoBERTa require GPU (8+ GB VRAM recommended)
- If no GPU: use Google Colab or Kaggle Notebooks
- Apple Silicon MPS is supported (used in current RoBERTa training)

## Git Workflow

- Large model files (*.pkl, *.pth) are gitignored
- Do not commit models over 100 MB (GitHub limit)
- Recent commits removed large .pkl files from history using git-filter-repo
- Force push required after history rewrite

## Current Status (2026-01-19)

- ✅ TF-IDF baseline: 86.58% accuracy
- ✅ Linguistic features: 65.00% accuracy
- 🚧 RoBERTa training: In progress (~8-9 hours expected)
- ⏳ Error analysis: Pending
- ⏳ BERT interpretation: Pending

## Conference Details

- **Conference**: Dialogue 2026
- **Deadline**: Usually April-May (check https://dialogue-conf.org/)
- **Format**: 8-12 pages, LaTeX template
- **Language**: English
