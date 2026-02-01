#!/usr/bin/env python3
"""
Update BERT predictions in error analysis with new cleaned model

Loads the existing full_predictions.csv, recomputes BERT predictions
using the new bert_category_classifier_cleaned model, and saves updated file.
"""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm.auto import tqdm
import joblib
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'error_analysis')

# Model paths
BERT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'bert_category_classifier_cleaned')
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, 'models', 'bert_label_encoder_cleaned.pkl')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'full_predictions.csv')
BACKUP_FILE = os.path.join(RESULTS_DIR, 'full_predictions_old_bert.csv')

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
    print("Updating BERT predictions with new cleaned model")
    print("="*80)

    # 1. Load existing predictions
    print("\n[1/5] Loading existing predictions...")
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"  Loaded {len(df)} predictions")

    # Backup old file
    print(f"\n[2/5] Creating backup...")
    df.to_csv(BACKUP_FILE, index=False)
    print(f"  Backup saved to: {BACKUP_FILE}")

    # 3. Load new BERT model
    print(f"\n[3/5] Loading new BERT model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH).to(device)
    model.eval()

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    categories = label_encoder.classes_
    print(f"  Categories: {categories.tolist()}")

    # 4. Recompute BERT predictions
    print(f"\n[4/5] Recomputing BERT predictions...")

    # Prepare dataset
    texts = df['text_preview'].tolist()

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Tokenize
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

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Create dataloader
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

    # Get predictions
    all_predictions = []

    for batch in tqdm(dataloader, desc="Computing BERT predictions"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())

    # Decode labels
    bert_preds = label_encoder.inverse_transform(all_predictions)

    # Update dataframe
    df['bert_pred'] = bert_preds
    df['bert_correct'] = df['true_category'] == df['bert_pred']

    # Recompute agreement columns
    df['all_three_agree'] = (
        (df['tfidf_pred'] == df['linguistic_pred']) &
        (df['linguistic_pred'] == df['bert_pred'])
    )

    df['all_correct'] = (
        df['tfidf_correct'] & df['linguistic_correct'] & df['bert_correct']
    )

    df['all_wrong'] = (
        (~df['tfidf_correct']) & (~df['linguistic_correct']) & (~df['bert_correct'])
    )

    # Calculate statistics
    bert_accuracy = (df['bert_correct'].sum() / len(df)) * 100

    print(f"\n  New BERT accuracy: {bert_accuracy:.2f}%")
    print(f"  Correct: {df['bert_correct'].sum()}/{len(df)}")

    # 5. Save updated predictions
    print(f"\n[5/5] Saving updated predictions...")
    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"  Saved to: {PREDICTIONS_FILE}")

    # Summary
    print("\n" + "="*80)
    print("UPDATE COMPLETE")
    print("="*80)
    print(f"\nOld file backed up to: {BACKUP_FILE}")
    print(f"Updated predictions: {PREDICTIONS_FILE}")
    print(f"\nNew BERT accuracy: {bert_accuracy:.2f}%")

    # Model agreement
    tfidf_bert_agree = (df['tfidf_pred'] == df['bert_pred']).sum() / len(df) * 100
    print(f"TF-IDF ↔ BERT agreement: {tfidf_bert_agree:.2f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
