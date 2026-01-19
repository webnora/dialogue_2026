#!/usr/bin/env python3
"""
BERT Fine-tuning Script for Guardian Genre Classification
Training BERT-base-uncased on cleaned_combined_guardian.csv

Expected accuracy: ~87-88%
"""

import pandas as pd
import torch
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
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

# Training hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0

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
print(f"BERT Fine-tuning for Guardian Genre Classification")
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
    print("[1/8] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Initial dataset size: {len(df)}")

    # 2. Clean data
    print("[2/8] Cleaning data...")
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
    print(f"  Categories: {df['category'].unique().tolist()}")

    # 3. Prepare texts and labels
    print("[3/8] Preparing texts and labels...")
    texts = df["cleaned_text"].tolist()
    categories = df["category"].tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)

    # Save label encoder
    os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"  Saved label encoder to {LABEL_ENCODER_PATH}")

    # 4. Split data
    print("[4/8] Splitting data...")
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

    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")
    print(f"  Test: {len(test_texts)}")

    # Validate texts
    validate_texts(train_texts)
    validate_texts(val_texts)
    validate_texts(test_texts)

    # 5. Create datasets
    print("[5/8] Creating datasets...")
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # 6. Tokenize
    print("[6/8] Tokenizing...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 7. Create dataloaders
    print("[7/8] Creating dataloaders...")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    train_dataloader = DataLoader(
        tokenized_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    val_dataloader = DataLoader(
        tokenized_val,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_test,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    # 8. Model
    print("\n[8/8] Training model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_encoder.classes_)
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    print(f"\nTraining configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total steps: {total_steps}")
    print(f"\n{'='*80}\n")

    # Training loop
    best_val_accuracy = 0
    training_history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        model.train()
        train_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Training")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{np.mean(train_losses[-100:]):.4f}"
            })

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_predictions = []
        val_true = []

        for batch in tqdm(val_dataloader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            val_predictions.extend(predictions.cpu().numpy())
            val_true.extend(batch["labels"].cpu().numpy())

        val_accuracy = accuracy_score(val_true, val_predictions)
        val_f1 = f1_score(val_true, val_predictions, average='macro')

        print(f"\n  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1 (macro): {val_f1:.4f}")

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1
        })

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  ✓ New best model! Saving...")
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Testing
    print(f"\n{'='*80}")
    print("Testing on hold-out set...")
    print(f"{'='*80}\n")

    model.load_state_dict(
        torch.load(f"{MODEL_SAVE_PATH}/pytorch_model.bin", map_location=device)
    )
    model.eval()

    test_predictions = []
    test_true = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        test_predictions.extend(predictions.cpu().numpy())
        test_true.extend(batch["labels"].cpu().numpy())

    test_accuracy = accuracy_score(test_true, test_predictions)
    test_f1_macro = f1_score(test_true, test_predictions, average='macro')
    test_f1_weighted = f1_score(test_true, test_predictions, average='weighted')

    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 (macro): {test_f1_macro:.4f}")
    print(f"  F1 (weighted): {test_f1_weighted:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(test_true, test_predictions)

    # Save results
    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CM_SAVE_PATH), exist_ok=True)

    metrics = {
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "best_val_accuracy": best_val_accuracy,
        "training_history": training_history,
        "categories": label_encoder.classes_.tolist(),
        "hyperparameters": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE
        }
    }

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    np.save(CM_SAVE_PATH, cm)

    print(f"\n{'='*80}")
    print(f"Results saved:")
    print(f"  Model: {MODEL_SAVE_PATH}/")
    print(f"  Metrics: {METRICS_SAVE_PATH}")
    print(f"  Confusion Matrix: {CM_SAVE_PATH}")
    print(f"  Label Encoder: {LABEL_ENCODER_PATH}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
