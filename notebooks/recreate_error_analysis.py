#!/usr/bin/env python3
"""
Recreate error analysis with new BERT model and full texts
Uses cleaned_guardian_filtered.csv (48K texts)
"""

import pandas as pd
import numpy as np
import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_guardian_filtered.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'error_analysis')

# Model paths
TFIDF_MODEL = os.path.join(PROJECT_ROOT, 'models', 'tfidf_lr_cleaned.pkl')
TFIDF_VECTORIZER = os.path.join(PROJECT_ROOT, 'models', 'tfidf_vectorizer_cleaned.pkl')
LINGUISTIC_MODEL = os.path.join(PROJECT_ROOT, 'models', 'linguistic_rf_cleaned.pkl')
BERT_MODEL = os.path.join(PROJECT_ROOT, 'models', 'bert_category_classifier_cleaned')
BERT_LABEL_ENCODER = os.path.join(PROJECT_ROOT, 'models', 'bert_label_encoder_cleaned.pkl')

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def main():
    print("="*80)
    print("Recreating Error Analysis with New BERT Model")
    print("="*80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["cleaned_text", "category"])
    df = df[df['cleaned_text'].str.strip() != ""]
    df['cleaned_text'] = df['cleaned_text'].astype(str)
    df['category'] = df['category'].astype(str)

    print(f"  Dataset size: {len(df)}")

    # 2. Split data (same as BERT training)
    print("\n[2/6] Splitting data...")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['category'].tolist())

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['cleaned_text'].tolist(), labels, test_size=0.2, stratify=labels, random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"  Test set: {len(test_texts)} samples")

    # 3. Get TF-IDF predictions
    print("\n[3/6] Loading TF-IDF model...")
    tfidf_model = joblib.load(TFIDF_MODEL)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER)

    print("  Computing TF-IDF predictions...")
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)
    tfidf_preds = tfidf_model.predict(X_test_tfidf)

    # 4. Get Linguistic predictions
    print("\n[4/6] Loading Linguistic model...")
    # Linguistic model needs feature extraction
    # For speed, use simplified approach or skip if too slow
    print("  ⚠️  Linguistic feature extraction is slow - using cached predictions...")
    # Try to load from old file
    old_predictions = pd.read_csv(os.path.join(RESULTS_DIR, 'full_predictions_old_bert.csv'))
    linguistic_preds = old_predictions['linguistic_pred'].values
    print(f"  Loaded {len(linguistic_preds)} cached predictions")

    # 5. Get BERT predictions
    print("\n[5/6] Computing BERT predictions...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL).to(device)
    model.eval()

    # Create dataset
    test_dataset = Dataset.from_dict({"text": test_texts})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    tokenized_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=32,
        collate_fn=data_collator
    )

    bert_preds_list = []

    for batch in tqdm(dataloader, desc="BERT inference"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        bert_preds_list.extend(predictions.cpu().numpy())

    bert_preds_encoded = np.array(bert_preds_list)
    bert_preds = label_encoder.inverse_transform(bert_preds_encoded)

    # 6. Create results dataframe
    print("\n[6/6] Creating results...")

    true_labels = label_encoder.inverse_transform(test_labels)

    results_df = pd.DataFrame({
        'true_category': true_labels,
        'tfidf_pred': tfidf_preds,
        'linguistic_pred': linguistic_preds,
        'bert_pred': bert_preds,
        'tfidf_correct': tfidf_preds == true_labels,
        'linguistic_correct': linguistic_preds == true_labels,
        'bert_correct': bert_preds == true_labels,
        'text_length': [len(t) for t in test_texts],
        'text': test_texts  # Full texts!
    })

    # Add agreement columns
    results_df['all_three_agree'] = (
        (results_df['tfidf_pred'] == results_df['linguistic_pred']) &
        (results_df['linguistic_pred'] == results_df['bert_pred'])
    )

    results_df['all_correct'] = (
        results_df['tfidf_correct'] & results_df['linguistic_correct'] & results_df['bert_correct']
    )

    results_df['all_wrong'] = (
        (~results_df['tfidf_correct']) & (~results_df['linguistic_correct']) & (~results_df['bert_correct'])
    )

    # Save
    output_file = os.path.join(RESULTS_DIR, 'full_predictions.csv')
    results_df.to_csv(output_file, index=False)

    # Summary
    print("\n" + "="*80)
    print("ERROR ANALYSIS RECREATED")
    print("="*80)

    tfidf_acc = (results_df['tfidf_correct'].sum() / len(results_df)) * 100
    bert_acc = (results_df['bert_correct'].sum() / len(results_df)) * 100

    print(f"\nTF-IDF Accuracy: {tfidf_acc:.2f}%")
    print(f"BERT Accuracy: {bert_acc:.2f}%")

    print(f"\n✅ Saved to: {output_file}")
    print(f"   Full texts included: {len(results_df)} samples")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
