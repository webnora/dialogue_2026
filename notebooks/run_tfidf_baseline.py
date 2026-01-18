#!/usr/bin/env python3
"""
Task 1.1: TF–IDF + Logistic Regression Baseline

Expected performance: Accuracy ~70-75%, Macro F1 ~0.68-0.73
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Get project root directory
project_root = os.path.dirname(os.path.dirname(__file__))

# Create results directory if it doesn't exist
results_dir = os.path.join(project_root, 'results')
models_dir = os.path.join(project_root, 'models')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("="*80)
print("Task 1.1: TF–IDF + Logistic Regression Baseline")
print("="*80)

# ============================================================================
# 1. Load and Explore Data
# ============================================================================
print("\n[1/15] Loading data...")
# Use absolute path to data file
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_combined_guardian.csv')
df = pd.read_csv(data_path)
print(f"✓ Dataset shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================================
# 2. Analyze Categories
# ============================================================================
print("\n[2/15] Analyzing categories...")
print(f"  Unique categories: {df['category'].nunique()}")
print(f"  Category distribution:")
category_counts = df['category'].value_counts()
for cat, count in category_counts.items():
    print(f"    {cat}: {count}")

# ============================================================================
# 3. Check Missing Values
# ============================================================================
print("\n[3/15] Checking missing values...")
missing_category = df['category'].isnull().sum()
missing_text = df['cleaned_text'].isnull().sum()
print(f"  Missing in category: {missing_category}")
print(f"  Missing in cleaned_text: {missing_text}")

# ============================================================================
# 4. Text Length Statistics
# ============================================================================
print("\n[4/15] Computing text length statistics...")
df['text_length'] = df['cleaned_text'].str.len()
print(f"  Min length: {df['text_length'].min()}")
print(f"  Max length: {df['text_length'].max()}")
print(f"  Mean length: {df['text_length'].mean():.0f}")
print(f"  Median length: {df['text_length'].median():.0f}")

# ============================================================================
# 5. Data Cleaning
# ============================================================================
print("\n[5/15] Cleaning data...")
original_size = len(df)
df = df.dropna(subset=['category', 'cleaned_text'])
print(f"  After removing NaN: {len(df)} (removed {original_size - len(df)})")

df = df[df['cleaned_text'].str.len() > 50]
print(f"  After removing short texts (<50 chars): {len(df)}")

df = df[df['cleaned_text'].str.len() < 20000]
print(f"  After removing long texts (>20000 chars): {len(df)}")

print(f"\n  Final category distribution:")
for cat, count in df['category'].value_counts().items():
    print(f"    {cat}: {count}")

# ============================================================================
# 6. Prepare Data for Training
# ============================================================================
print("\n[6/15] Preparing data for training...")
texts = df['cleaned_text'].tolist()
labels = df['category'].tolist()
print(f"  Total samples: {len(texts)}")
print(f"  Unique labels: {set(labels)}")

# ============================================================================
# 7. Split Data (Train/Val/Test)
# ============================================================================
print("\n[7/15] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")

# ============================================================================
# 8. Save Data Splits
# ============================================================================
print("\n[8/15] Saving data splits...")
splits = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test
}
joblib.dump(splits, results_dir + '/data_splits.pkl')
print("  ✓ Saved to results_dir + '/'data_splits.pkl")

# ============================================================================
# 9. Initialize TF–IDF Vectorizer
# ============================================================================
print("\n[9/15] Initializing TF–IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    stop_words='english'
)
print("  Parameters:")
print(f"    max_features: 10000")
print(f"    ngram_range: (1, 2)")
print(f"    min_df: 5")
print(f"    max_df: 0.8")

# ============================================================================
# 10. Fit and Transform
# ============================================================================
print("\n[10/15] Fitting vectorizer and transforming data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"  Training set shape: {X_train_tfidf.shape}")
print(f"  Validation set shape: {X_val_tfidf.shape}")
print(f"  Test set shape: {X_test_tfidf.shape}")

# ============================================================================
# 11. Grid Search
# ============================================================================
print("\n[11/15] Performing Grid Search with 5-fold CV...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

lr_base = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='multinomial'
)

grid_search = GridSearchCV(
    lr_base, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_tfidf, y_train)

print(f"\n  ✓ Best parameters: {grid_search.best_params_}")
print(f"  ✓ Best CV score: {grid_search.best_score_:.4f}")

# ============================================================================
# 12. Predictions
# ============================================================================
print("\n[12/15] Making predictions...")
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train_tfidf)
y_val_pred = best_model.predict(X_val_tfidf)
y_test_pred = best_model.predict(X_test_tfidf)

print("  ✓ Predictions completed for all sets")

# ============================================================================
# 13. Evaluate Performance
# ============================================================================
print("\n[13/15] Evaluating model performance...")

# Training
train_acc = accuracy_score(y_train, y_train_pred)
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')

# Validation
val_acc = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')

# Test
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
# 14. Confusion Matrix
# ============================================================================
print("\n[14/15] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred)
class_labels = best_model.classes_

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Test Set (TF–IDF + LR)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(results_dir + '/tfidf_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved confusion matrix visualization to results_dir + '/'tfidf_confusion_matrix.png")

# Save confusion matrix
joblib.dump(cm, results_dir + '/tfidf_confusion_matrix.npy')
print("  ✓ Saved confusion matrix to results_dir + '/'tfidf_confusion_matrix.npy")

# ============================================================================
# 15. Extract Lexical Markers
# ============================================================================
print("\n[15/15] Extracting lexical markers for each genre...")
feature_names = vectorizer.get_feature_names_out()

lexical_markers = {}
for i, genre in enumerate(class_labels):
    coef = best_model.coef_[i]
    top_indices = np.argsort(coef)[-50:][::-1]
    lexical_markers[genre] = [
        (feature_names[idx], coef[idx])
        for idx in top_indices
    ]

joblib.dump(lexical_markers, results_dir + '/tfidf_lexical_markers.pkl')
print("  ✓ Saved lexical markers to results_dir + '/'tfidf_lexical_markers.pkl")

# Display top 10 words for each genre
print("\n  Top 10 words for each genre:")
for genre, markers in lexical_markers.items():
    print(f"\n    {genre.upper()}:")
    for word, weight in markers[:10]:
        print(f"      {word:20s} {weight:8.4f}")

# ============================================================================
# Save Model and Metrics
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save vectorizer
joblib.dump(vectorizer, models_dir + '/tfidf_vectorizer.pkl')
print("✓ Vectorizer saved to models_dir + '/'tfidf_vectorizer.pkl")

# Save model
joblib.dump(best_model, models_dir + '/tfidf_lr.pkl')
print("✓ Model saved to models_dir + '/'tfidf_lr.pkl")

# Save metrics
metrics = {
    'model': 'TF–IDF + Logistic Regression',
    'best_params': {
        'C': int(grid_search.best_params_['C']),
        'penalty': grid_search.best_params_['penalty'],
        'solver': grid_search.best_params_['solver']
    },
    'train_accuracy': float(train_acc),
    'train_f1_macro': float(train_f1_macro),
    'val_accuracy': float(val_acc),
    'val_f1_macro': float(val_f1_macro),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'cv_score': float(grid_search.best_score_)
}

with open(results_dir + '/tfidf_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Metrics saved to results_dir + '/'tfidf_metrics.json")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nModel: TF–IDF + Logistic Regression")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"\n--- Test Set Performance ---")
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Macro F1:  {test_f1_macro:.4f}")
print(f"\n--- Expected vs Actual ---")
print(f"Expected Accuracy: ~70-75%")
print(f"Actual Accuracy:   {test_acc*100:.2f}%")
print(f"\nExpected Macro F1: ~0.68-0.73")
print(f"Actual Macro F1:   {test_f1_macro:.4f}")

if test_acc >= 0.70 and test_acc <= 0.75:
    print("\n✓ Accuracy within expected range!")
elif test_acc > 0.75:
    print("\n✓ Accuracy exceeds expectations!")
else:
    print("\n⚠ Accuracy below expected range")

print("\n" + "="*80)
print("TASK 1.1 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  - models_dir + '/'tfidf_vectorizer.pkl")
print("  - models_dir + '/'tfidf_lr.pkl")
print("  - results_dir + '/'tfidf_metrics.json")
print("  - results_dir + '/'tfidf_confusion_matrix.npy")
print("  - results_dir + '/'tfidf_confusion_matrix.png")
print("  - results_dir + '/'tfidf_lexical_markers.pkl")
print("  - results_dir + '/'data_splits.pkl")
