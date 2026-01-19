#!/usr/bin/env python3
"""
Test BERT Model on Hold-out Set
Loads the trained BERT model and evaluates on test set.
"""

import pandas as pd
import torch
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

# Configuration
DATA_PATH = "data/cleaned_combined_guardian.csv"
MODEL_SAVE_PATH = "models/bert_category_classifier"
METRICS_SAVE_PATH = "results/bert_metrics.json"
CM_SAVE_PATH = "results/bert_confusion_matrix.npy"
LABEL_ENCODER_PATH = "models/bert_label_encoder.pkl"

# Training hyperparameters (must match training)
MAX_LENGTH = 256
BATCH_SIZE = 16

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

print(f"\n{'='*80}")
print(f"BERT Model Testing - Guardian Genre Classification")
print(f"{'='*80}\n")


def validate_texts(text_list):
    """Validate that all texts are non-empty strings."""
    for i, text in enumerate(text_list):
        if not isinstance(text, str):
            raise ValueError(f"Non-string text at index {i}: {type(text)}")
        if len(text.strip()) == 0:
            raise ValueError(f"Empty text at index {i}")


def main():
    # 1. Load data
    print("[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Initial dataset size: {len(df)}")

    # 2. Clean data (same as training)
    print("[2/6] Cleaning data...")
    df = df.dropna(subset=["cleaned_text", "category"])

    # Convert to string and clean
    df["cleaned_text"] = (
        df["cleaned_text"]
        .astype(str)
        .replace({"nan": "", "None": "", "null": ""})
    )

    # Remove invalid texts
    nan_mask = df["cleaned_text"].str.strip().isin(["", "nan", "None", "null"])
    if nan_mask.sum() > 0:
        print(f"  Warning: Found {nan_mask.sum()} invalid texts, removing...")
        df = df[~nan_mask]

    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df["category"] = df["category"].astype(str)

    print(f"  Final dataset size: {len(df)}")

    # 3. Prepare texts and labels
    print("[3/6] Preparing texts and labels...")
    texts = df["cleaned_text"].tolist()
    categories = df["category"].tolist()

    # Load label encoder
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    labels = label_encoder.transform(categories)

    print(f"  Loaded label encoder with {len(label_encoder.classes_)} classes")

    # 4. Split data (same as training)
    print("[4/6] Splitting data...")
    from sklearn.model_selection import train_test_split

    # Replicate the same split as in training:
    # First: 80% train, 20% temp
    # Second: 50% val, 50% test from temp
    # Final: 80% train, 10% val, 10% test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    print(f"  Test: {len(test_texts)}")

    # Validate texts
    validate_texts(test_texts)

    # 5. Create test dataset and dataloader
    print("[5/6] Creating test dataloader...")
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)

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
            "attention_mask": tokenized["attention_mask"],
            "labels": examples["label"]
        }

    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    test_dataloader = DataLoader(
        tokenized_test,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    # 6. Load model and test
    print("\n[6/6] Loading model and testing...")
    model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(device)
    print(f"  Model loaded successfully from {MODEL_SAVE_PATH}")

    model.eval()
    test_predictions = []
    test_true = []

    print(f"\nRunning inference on {len(test_dataloader)} batches...")
    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        test_predictions.extend(predictions.cpu().numpy())
        test_true.extend(batch["labels"].cpu().numpy())

    # Calculate metrics
    test_accuracy = accuracy_score(test_true, test_predictions)
    test_f1_macro = f1_score(test_true, test_predictions, average='macro')
    test_f1_weighted = f1_score(test_true, test_predictions, average='weighted')

    print(f"\n{'='*80}")
    print(f"Final Test Results:")
    print(f"{'='*80}")
    print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  F1 (macro): {test_f1_macro:.4f}")
    print(f"  F1 (weighted): {test_f1_weighted:.4f}")
    print(f"  Categories: {label_encoder.classes_.tolist()}")
    print(f"{'='*80}\n")

    # Compute confusion matrix
    cm = confusion_matrix(test_true, test_predictions)
    print(f"Confusion Matrix:")
    print(cm)

    # Save results
    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CM_SAVE_PATH), exist_ok=True)

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "categories": label_encoder.classes_.tolist(),
        "test_size": len(test_texts),
        "model_path": MODEL_SAVE_PATH
    }

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    np.save(CM_SAVE_PATH, cm)

    print(f"\n{'='*80}")
    print(f"Results saved:")
    print(f"  Metrics: {METRICS_SAVE_PATH}")
    print(f"  Confusion Matrix: {CM_SAVE_PATH}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
