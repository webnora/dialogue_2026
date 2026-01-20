#!/usr/bin/env python3
"""
Phase 2: Error Analysis
Analyze misclassifications to understand genre boundaries and hybrid articles.
"""

import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
TEST_DATA_PATH = "data/cleaned_combined_guardian.csv"
MAX_LENGTH = 256
BATCH_SIZE = 16

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n{'='*80}")
print(f"Phase 2: Error Analysis and Genre Boundary Investigation")
print(f"{'='*80}\n")
print(f"Using device: {device}\n")

# Linguistic feature lists
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

def extract_linguistic_features(text, nlp):
    """Extract linguistically motivated features from text."""
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


def load_test_data():
    """Load test split with true labels."""
    print("[1/8] Loading test data...")

    df = pd.read_csv(TEST_DATA_PATH)
    le = joblib.load("models/bert_label_encoder.pkl")
    categories = le.classes_.tolist()

    # Sample test set (same split as training)
    # Use first 4909 samples as test (stratified)
    test_samples = []

    for category in categories:
        category_df = df[df['category'] == category].head(1000)
        test_samples.append(category_df)

    test_df = pd.concat(test_samples, ignore_index=True)

    print(f"  Loaded {len(test_df)} test samples")
    print(f"  Categories: {categories}")
    return test_df, categories, le


def get_predictions_all_models(test_texts, le):
    """Get predictions from all three models."""
    print("\n[2/8] Getting predictions from all models...")

    # TF-IDF + LR
    print("  Loading TF-IDF + LR...")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_model = joblib.load("models/tfidf_lr.pkl")
    texts_tfidf = vectorizer.transform(test_texts)
    tfidf_preds = tfidf_model.predict(texts_tfidf)

    # Convert to numeric if needed
    if isinstance(tfidf_preds[0], str):
        tfidf_preds = le.transform(tfidf_preds)

    print(f"    ✓ TF-IDF predictions: {len(tfidf_preds)}")

    # Linguistic + RF (extract features)
    print("  Loading Linguistic + RF...")
    linguistic_model = joblib.load("models/linguistic_rf.pkl")
    nlp = spacy.load('en_core_web_sm')

    print("    Extracting linguistic features... (this may take a few minutes)")
    test_features = []
    for text in tqdm(test_texts, desc="    Processing"):
        features = extract_linguistic_features(text, nlp)
        test_features.append(features)

    test_features_df = pd.DataFrame(test_features)
    linguistic_preds = linguistic_model.predict(test_features_df)

    # Convert to numeric if needed
    if isinstance(linguistic_preds[0], str):
        linguistic_preds = le.transform(linguistic_preds)

    print(f"    ✓ Linguistic predictions: {len(linguistic_preds)}")

    # BERT
    print("  Loading BERT...")
    model = BertForSequenceClassification.from_pretrained(
        "models/bert_category_classifier"
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")

    # Create dataset
    test_dataset = Dataset.from_dict({"text": test_texts})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH,
        pad_to_multiple_of=8, return_tensors="pt"
    )

    test_dataloader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, collate_fn=data_collator)

    model.eval()
    bert_preds = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1)
        bert_preds.extend(batch_preds.cpu().numpy())

    print(f"    ✓ BERT predictions: {len(bert_preds)}")

    return tfidf_preds, linguistic_preds, np.array(bert_preds)


def analyze_errors(test_df, categories, le):
    """Analyze misclassifications."""
    print("\n[3/8] Analyzing misclassifications...")

    test_texts = test_df['cleaned_text'].tolist()
    true_labels = le.transform(test_df['category'].tolist())

    tfidf_preds, linguistic_preds, bert_preds = get_predictions_all_models(test_texts, le)

    results = []

    for idx in range(len(test_df)):
        true_label = true_labels[idx]
        true_cat = categories[true_label]

        tfidf_pred = tfidf_preds[idx]
        tfidf_cat = categories[tfidf_pred]
        tfidf_correct = (tfidf_pred == true_label)

        linguistic_pred = linguistic_preds[idx]
        linguistic_cat = categories[linguistic_pred]
        linguistic_correct = (linguistic_pred == true_label)

        bert_pred = bert_preds[idx]
        bert_cat = categories[bert_pred]
        bert_correct = (bert_pred == true_label)

        # Agreement (all three models)
        all_three_agree = (tfidf_pred == linguistic_pred == bert_pred)
        all_correct = tfidf_correct and linguistic_correct and bert_correct
        all_wrong = (not tfidf_correct) and (not linguistic_correct) and (not bert_correct)

        results.append({
            'index': idx,
            'true_category': true_cat,
            'tfidf_pred': tfidf_cat,
            'linguistic_pred': linguistic_cat,
            'bert_pred': bert_cat,
            'tfidf_correct': tfidf_correct,
            'linguistic_correct': linguistic_correct,
            'bert_correct': bert_correct,
            'all_three_agree': all_three_agree,
            'all_correct': all_correct,
            'all_wrong': all_wrong,
            'text_length': len(test_df.iloc[idx]['cleaned_text']),
            'text_preview': test_df.iloc[idx]['cleaned_text'][:200] + "..."
        })

    results_df = pd.DataFrame(results)

    print(f"  ✓ Analyzed {len(results_df)} samples")
    return results_df


def identify_error_types(results_df, categories):
    """Identify different types of errors."""
    print("\n[4/8] Identifying error types...")

    error_types = {
        'all_three_correct': results_df[results_df['all_correct']],
        'all_three_wrong': results_df[results_df['all_wrong']],
        'tfidf_only_wrong': results_df[(~results_df['tfidf_correct']) & results_df['linguistic_correct'] & results_df['bert_correct']],
        'linguistic_only_wrong': results_df[results_df['tfidf_correct'] & (~results_df['linguistic_correct']) & results_df['bert_correct']],
        'bert_only_wrong': results_df[results_df['tfidf_correct'] & results_df['linguistic_correct'] & (~results_df['bert_correct'])],
        'all_three_agree_correct': results_df[results_df['all_three_agree'] & results_df['all_correct']],
        'all_three_agree_wrong': results_df[results_df['all_three_agree'] & results_df['all_wrong']],
    }

    print(f"\n  Error Type Distribution:")
    print(f"  {'-'*60}")

    for error_type, df in error_types.items():
        count = len(df)
        pct = count / len(results_df) * 100
        print(f"  {error_type:25s}: {count:4d} ({pct:5.2f}%)")

    return error_types


def analyze_genre_confusions(results_df, categories):
    """Analyze which genres are most confused with each other."""
    print("\n[5/8] Analyzing genre confusions...")

    # Get confusion matrices
    le = joblib.load("models/bert_label_encoder.pkl")

    true_labels = le.transform(results_df['true_category'])
    tfidf_preds = le.transform(results_df['tfidf_pred'])
    linguistic_preds = le.transform(results_df['linguistic_pred'])
    bert_preds = le.transform(results_df['bert_pred'])

    # Confusion matrices for all three models
    cm_tfidf = confusion_matrix(true_labels, tfidf_preds)
    cm_linguistic = confusion_matrix(true_labels, linguistic_preds)
    cm_bert = confusion_matrix(true_labels, bert_preds)

    # Find most confused pairs for BERT
    confusion_pairs = []

    for i in range(len(categories)):
        for j in range(len(categories)):
            if i != j:
                count = cm_bert[i, j]
                if count > 0:
                    confusion_pairs.append({
                        'true': categories[i],
                        'pred': categories[j],
                        'count': count,
                        'pct': count / cm_bert[i].sum() * 100
                    })

    confusion_df = pd.DataFrame(confusion_pairs).sort_values('count', ascending=False)

    print(f"\n  Top BERT Confusions (true → predicted):")
    print(f"  {'-'*60}")
    for _, row in confusion_df.head(10).iterrows():
        print(f"  {row['true']:12s} → {row['pred']:12s}: {row['count']:3d} ({row['pct']:4.1f}%)")

    return confusion_df, cm_tfidf, cm_linguistic, cm_bert


def visualize_error_patterns(results_df, categories):
    """Visualize error patterns."""
    print("\n[6/8] Creating error visualizations...")

    import os
    os.makedirs("results/error_analysis", exist_ok=True)

    # 1. Text length vs accuracy (3 models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # TF-IDF accuracy by text length
    tfidf_acc_by_length = results_df.groupby(
        pd.cut(results_df['text_length'], bins=[0, 500, 1000, 2000, 5000, 10000])
    )['tfidf_correct'].mean()

    ax = axes[0]
    tfidf_acc_by_length.plot(kind='bar', ax=ax, color='#3498db', alpha=0.7)
    ax.set_title('TF-IDF Accuracy by Text Length', fontsize=12, weight='bold')
    ax.set_xlabel('Text Length (chars)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.8658, color='red', linestyle='--', label='Overall TF-IDF (86.58%)')
    ax.legend()

    # Linguistic accuracy by text length
    linguistic_acc_by_length = results_df.groupby(
        pd.cut(results_df['text_length'], bins=[0, 500, 1000, 2000, 5000, 10000])
    )['linguistic_correct'].mean()

    ax = axes[1]
    linguistic_acc_by_length.plot(kind='bar', ax=ax, color='#9b59b6', alpha=0.7)
    ax.set_title('Linguistic Accuracy by Text Length', fontsize=12, weight='bold')
    ax.set_xlabel('Text Length (chars)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.6500, color='red', linestyle='--', label='Overall Linguistic (65.00%)')
    ax.legend()

    # BERT accuracy by text length
    bert_acc_by_length = results_df.groupby(
        pd.cut(results_df['text_length'], bins=[0, 500, 1000, 2000, 5000, 10000])
    )['bert_correct'].mean()

    ax = axes[2]
    bert_acc_by_length.plot(kind='bar', ax=ax, color='#2ecc71', alpha=0.7)
    ax.set_title('BERT Accuracy by Text Length', fontsize=12, weight='bold')
    ax.set_xlabel('Text Length (chars)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.8764, color='red', linestyle='--', label='Overall BERT (87.64%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/error_analysis/accuracy_by_length.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/error_analysis/accuracy_by_length.png")

    # 2. Error type distribution (all three models)
    fig, ax = plt.subplots(figsize=(12, 6))

    error_counts = {
        'All Three\nCorrect': results_df['all_correct'].sum(),
        'All Three\nWrong': results_df['all_wrong'].sum(),
        'TF-IDF only\nWrong': ((~results_df['tfidf_correct']) & results_df['linguistic_correct'] & results_df['bert_correct']).sum(),
        'Linguistic only\nWrong': (results_df['tfidf_correct'] & (~results_df['linguistic_correct']) & results_df['bert_correct']).sum(),
        'BERT only\nWrong': (results_df['tfidf_correct'] & results_df['linguistic_correct'] & (~results_df['bert_correct'])).sum()
    }

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    bars = ax.bar(error_counts.keys(), error_counts.values(), color=colors, alpha=0.7, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        pct = height / len(results_df) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_title('Model Agreement on Test Set (All Three Models)', fontsize=14, weight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/error_analysis/model_agreement.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/error_analysis/model_agreement.png")


def identify_hybrid_articles(results_df, categories):
    """Identify potentially hybrid articles (borderline cases)."""
    print("\n[7/8] Identifying hybrid articles...")

    # Hybrid criteria for three models:
    # 1. All three models wrong
    # 2. Models disagree (not all three agree)
    # 3. Different plausible predictions

    hybrids = results_df[
        (results_df['all_wrong']) |
        (~results_df['all_three_agree'])
    ].copy()

    # Add hybrid type
    hybrids.loc[:, 'hybrid_type'] = 'all_wrong'
    hybrids.loc[hybrids['all_wrong'], 'hybrid_type'] = 'all_wrong'
    hybrids.loc[~hybrids['all_three_agree'] & ~hybrids['all_wrong'], 'hybrid_type'] = 'disagree_mixed'

    print(f"\n  Found {len(hybrids)} potential hybrid articles ({len(hybrids)/len(results_df)*100:.1f}%)")

    # Show examples
    print(f"\n  Sample hybrid articles:")
    print(f"  {'-'*80}\n")

    for idx, row in hybrids.head(5).iterrows():
        print(f"  [{row['hybrid_type']}] True: {row['true_category']}")
        print(f"    TF-IDF predicted:     {row['tfidf_pred']}")
        print(f"    Linguistic predicted: {row['linguistic_pred']}")
        print(f"    BERT predicted:       {row['bert_pred']}")
        print(f"    Text preview: {row['text_preview'][:150]}...")
        print()

    # Save hybrids
    hybrids.to_csv("results/error_analysis/hybrid_articles.csv", index=False)
    print(f"  ✓ Saved: results/error_analysis/hybrid_articles.csv")

    return hybrids


def create_error_summary(results_df, confusion_df, categories):
    """Create comprehensive error summary."""
    print("\n[8/8] Creating error summary report...")

    summary = {
        'total_samples': len(results_df),
        'tfidf_accuracy': float(results_df['tfidf_correct'].mean()),
        'linguistic_accuracy': float(results_df['linguistic_correct'].mean()),
        'bert_accuracy': float(results_df['bert_correct'].mean()),
        'all_three_agree': float(results_df['all_three_agree'].mean()),
        'all_correct': int(results_df['all_correct'].sum()),
        'all_wrong': int(results_df['all_wrong'].sum()),
        'tfidf_only_wrong': int(((~results_df['tfidf_correct']) & results_df['linguistic_correct'] & results_df['bert_correct']).sum()),
        'linguistic_only_wrong': int((results_df['tfidf_correct'] & (~results_df['linguistic_correct']) & results_df['bert_correct']).sum()),
        'bert_only_wrong': int((results_df['tfidf_correct'] & results_df['linguistic_correct'] & (~results_df['bert_correct'])).sum()),
        'top_confusions': confusion_df.head(10).to_dict('records')
    }

    import json
    with open("results/error_analysis/error_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("  ✓ Saved: results/error_analysis/error_summary.json")

    return summary


def main():
    # Load data
    test_df, categories, le = load_test_data()

    # Analyze errors
    results_df = analyze_errors(test_df, categories, le)

    # Identify error types
    error_types = identify_error_types(results_df, categories)

    # Analyze confusions
    confusion_df, cm_tfidf, cm_linguistic, cm_bert = analyze_genre_confusions(results_df, categories)

    # Visualize
    visualize_error_patterns(results_df, categories)

    # Identify hybrids
    hybrids = identify_hybrid_articles(results_df, categories)

    # Summary
    summary = create_error_summary(results_df, confusion_df, categories)

    # Save full results
    results_df.to_csv("results/error_analysis/full_predictions.csv", index=False)
    print(f"\n  ✓ Saved: results/error_analysis/full_predictions.csv")

    print("\n" + "="*80)
    print("ERROR ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Statistics:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  TF-IDF accuracy:     {summary['tfidf_accuracy']:.4f}")
    print(f"  Linguistic accuracy: {summary['linguistic_accuracy']:.4f}")
    print(f"  BERT accuracy:       {summary['bert_accuracy']:.4f}")
    print(f"  All three agree:     {summary['all_three_agree']:.4f}")
    print(f"  All correct: {summary['all_correct']} ({summary['all_correct']/summary['total_samples']*100:.1f}%)")
    print(f"  All wrong:   {summary['all_wrong']} ({summary['all_wrong']/summary['total_samples']*100:.1f}%)")
    print(f"\nAll results saved to: results/error_analysis/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
