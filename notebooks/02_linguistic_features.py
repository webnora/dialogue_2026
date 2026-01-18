#!/usr/bin/env python3
"""
Task 1.2: Linguistic Features + Random Forest

Extract linguistically motivated features and train Random Forest classifier.
Expected performance: Accuracy ~78%, Macro F1 ~0.76
"""

import pandas as pd
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Get project root directory
project_root = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(project_root, 'results')
models_dir = os.path.join(project_root, 'models')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("="*80)
print("Task 1.2: Linguistic Features + Random Forest")
print("="*80)

# ============================================================================
# 1. Load Data Splits
# ============================================================================
print("\n[1/10] Loading data splits...")
splits = joblib.load(os.path.join(results_dir, 'data_splits.pkl'))

X_train = splits['X_train']
X_val = splits['X_val']
X_test = splits['X_test']
y_train = splits['y_train']
y_val = splits['y_val']
y_test = splits['y_test']

print(f"  Train: {len(X_train)} samples")
print(f"  Val: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")

# ============================================================================
# 2. Initialize spaCy
# ============================================================================
print("\n[2/10] Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')
print("  ✓ spaCy model loaded")

# ============================================================================
# 3. Define Linguistic Feature Extractors
# ============================================================================
print("\n[3/10] Defining feature extractors...")

# Lists for stance markers and hedges
STANCE_MARKERS = [
    'arguably', 'reportedly', 'seemingly', 'apparently',
    'undoubtedly', 'clearly', 'obviously', 'evidently',
    'supposedly', 'presumably', 'ostensibly', 'arguably'
]

HEDGES = [
    'perhaps', 'possibly', 'somewhat', 'rather',
    'quite', 'relatively', 'comparatively', 'somewhat'
]

MODAL_VERBS = [
    'can', 'could', 'may', 'might', 'must',
    'should', 'ought', 'would', 'shall'
]

REPORTING_VERBS = [
    'said', 'says', 'say', 'told', 'tells', 'tell',
    'claimed', 'claims', 'claim', 'stated', 'states', 'state',
    'reported', 'reports', 'report', 'announced', 'announces'
]

print("  ✓ Stance markers:", len(STANCE_MARKERS))
print("  ✓ Hedges:", len(HEDGES))
print("  ✓ Modal verbs:", len(MODAL_VERBS))
print("  ✓ Reporting verbs:", len(REPORTING_VERBS))

def extract_linguistic_features(text, nlp):
    """
    Extract linguistically motivated features from text.
    """
    doc = nlp(text)

    features = {}

    # [1] Type-Token Ratio (lexical diversity)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    if len(tokens) > 0:
        features['type_token_ratio'] = len(set(tokens)) / len(tokens)
    else:
        features['type_token_ratio'] = 0

    # [2] Average sentence length
    sentences = list(doc.sents)
    if len(sentences) > 0:
        sent_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths)
    else:
        features['avg_sentence_length'] = 0

    # [3-5] Pronoun ratios (1st, 2nd, 3rd person)
    first_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['i', 'we', 'me', 'us', 'my', 'our'])
    second_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['you', 'your'])
    third_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their'])

    total_tokens = len(doc)
    if total_tokens > 0:
        features['first_person_ratio'] = first_person / total_tokens
        features['second_person_ratio'] = second_person / total_tokens
        features['third_person_ratio'] = third_person / total_tokens
    else:
        features['first_person_ratio'] = 0
        features['second_person_ratio'] = 0
        features['third_person_ratio'] = 0

    # [6] Modal verb ratio
    modal_count = sum(1 for token in doc if token.text.lower() in MODAL_VERBS)
    features['modal_ratio'] = modal_count / total_tokens if total_tokens > 0 else 0

    # [7] Stance markers ratio
    stance_count = sum(1 for token in doc if token.text.lower() in STANCE_MARKERS)
    features['stance_markers_ratio'] = stance_count / total_tokens if total_tokens > 0 else 0

    # [8] Hedges ratio
    hedge_count = sum(1 for token in doc if token.text.lower() in HEDGES)
    features['hedges_ratio'] = hedge_count / total_tokens if total_tokens > 0 else 0

    # [9] Quotes ratio (approximate by counting quotation marks)
    quote_count = text.count('"')
    features['quotes_ratio'] = quote_count / total_tokens if total_tokens > 0 else 0

    # [10] Reporting verbs ratio
    reporting_count = sum(1 for token in doc if token.text.lower() in REPORTING_VERBS)
    features['reporting_verbs_ratio'] = reporting_count / total_tokens if total_tokens > 0 else 0

    return features

# ============================================================================
# 4. Extract Features from Training Set
# ============================================================================
print("\n[4/10] Extracting features from training set...")
print("  (This may take a few minutes...)")

X_train_features = []
for text in tqdm(X_train, desc="  Processing"):
    features = extract_linguistic_features(text, nlp)
    X_train_features.append(features)

# Convert to DataFrame
train_features_df = pd.DataFrame(X_train_features)
print(f"  ✓ Extracted {train_features_df.shape[1]} features from {len(X_train)} texts")

# ============================================================================
# 5. Extract Features from Validation Set
# ============================================================================
print("\n[5/10] Extracting features from validation set...")

X_val_features = []
for text in tqdm(X_val, desc="  Processing"):
    features = extract_linguistic_features(text, nlp)
    X_val_features.append(features)

val_features_df = pd.DataFrame(X_val_features)
print(f"  ✓ Extracted {val_features_df.shape[1]} features from {len(X_val)} texts")

# ============================================================================
# 6. Extract Features from Test Set
# ============================================================================
print("\n[6/10] Extracting features from test set...")

X_test_features = []
for text in tqdm(X_test, desc="  Processing"):
    features = extract_linguistic_features(text, nlp)
    X_test_features.append(features)

test_features_df = pd.DataFrame(X_test_features)
print(f"  ✓ Extracted {test_features_df.shape[1]} features from {len(X_test)} texts")

# ============================================================================
# 7. Grid Search for Random Forest
# ============================================================================
print("\n[7/10] Performing Grid Search for Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(train_features_df, y_train)

print(f"\n  ✓ Best parameters: {grid_search.best_params_}")
print(f"  ✓ Best CV score: {grid_search.best_score_:.4f}")

# ============================================================================
# 8. Train Final Model
# ============================================================================
print("\n[8/10] Training final model...")

best_model = grid_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(train_features_df)
y_val_pred = best_model.predict(val_features_df)
y_test_pred = best_model.predict(test_features_df)

print("  ✓ Predictions completed for all sets")

# ============================================================================
# 9. Evaluate Performance
# ============================================================================
print("\n[9/10] Evaluating model performance...")

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')

val_acc = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

print("\n  Training Set:")
print(f"    Accuracy: {train_acc:.4f}")
print(f"    Macro F1: {train_f1_macro:.4f}")

print("\n  Validation Set:")
print(f"    Accuracy: {val_acc:.4f}")
print(f"    Macro F1: {val_f1_macro:.4f}")

print("\n  Test Set:")
print(f"    Accuracy: {test_acc:.4f}")
print(f"    Macro F1: {test_f1_macro:.4f}")

# ============================================================================
# 10. Feature Importance Analysis
# ============================================================================
print("\n[10/10] Analyzing feature importance...")

feature_names = train_features_df.columns.tolist()
feature_importance = best_model.feature_importances_

# Sort features by importance
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n  Feature Importance (descending):")
print("  " + "-"*50)
for idx, row in feature_importance_df.iterrows():
    print(f"  {row['feature']:30s} {row['importance']:8.4f}")

# ============================================================================
# Confusion Matrix
# ============================================================================
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred)
class_labels = best_model.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Test Set (Linguistic Features + RF)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'linguistic_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("  ✓ Saved confusion matrix visualization")

joblib.dump(cm, os.path.join(results_dir, 'linguistic_confusion_matrix.npy'))
print("  ✓ Saved confusion matrix")

# ============================================================================
# Save Model and Results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save model
joblib.dump(best_model, os.path.join(models_dir, 'linguistic_rf.pkl'))
print("✓ Model saved to models/linguistic_rf.pkl")

# Save feature importances
joblib.dump(feature_importance_df, os.path.join(results_dir, 'linguistic_feature_importance.pkl'))
print("✓ Feature importance saved to results/linguistic_feature_importance.pkl")

# Save metrics
metrics = {
    'model': 'Linguistic Features + Random Forest',
    'best_params': {
        'n_estimators': int(grid_search.best_params_['n_estimators']),
        'max_depth': int(grid_search.best_params_['max_depth']) if grid_search.best_params_['max_depth'] else None,
        'min_samples_split': int(grid_search.best_params_['min_samples_split']),
        'min_samples_leaf': int(grid_search.best_params_['min_samples_leaf'])
    },
    'train_accuracy': float(train_acc),
    'train_f1_macro': float(train_f1_macro),
    'val_accuracy': float(val_acc),
    'val_f1_macro': float(val_f1_macro),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'cv_score': float(grid_search.best_score_),
    'n_features': int(train_features_df.shape[1])
}

with open(os.path.join(results_dir, 'linguistic_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Metrics saved to results/linguistic_metrics.json")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nModel: Linguistic Features + Random Forest")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Macro F1:  {test_f1_macro:.4f}")
print(f"\n--- Expected vs Actual ---")
print(f"Expected Accuracy: ~78%")
print(f"Actual Accuracy:   {test_acc*100:.2f}%")
print(f"\nExpected Macro F1: ~0.76")
print(f"Actual Macro F1:   {test_f1_macro:.4f}")

if test_acc >= 0.75 and test_acc <= 0.80:
    print("\n✓ Accuracy within expected range!")
elif test_acc > 0.80:
    print("\n✓ Accuracy exceeds expectations!")
else:
    print("\n⚠ Accuracy below expected range")

print("\n" + "="*80)
print("TASK 1.2 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  - models/linguistic_rf.pkl")
print("  - results/linguistic_metrics.json")
print("  - results/linguistic_confusion_matrix.npy")
print("  - results/linguistic_confusion_matrix.png")
print("  - results/linguistic_feature_importance.pkl")
