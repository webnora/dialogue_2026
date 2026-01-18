#!/usr/bin/env python3
"""
Task 1.3: RoBERTa Fine-tuning for Genre Classification

Запуск: python3 notebooks/run_roberta_training.py
Время выполнения: ~2-3 часа на MPS (Apple Silicon), ~6-8 часов на CPU

Ожидаемый результат: ~89% accuracy
"""

import pandas as pd
import torch
import numpy as np
import os
import json
import pickle
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')  # Для сохранения графиков без дисплея
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== КОНФИГУРАЦИЯ ====================
config = {
    "model_name": "roberta-base",
    "num_labels": 5,
    "max_length": 512,
    "batch_size": 16,
    "epochs": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "random_state": 42,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}

# ==================== УСТРОЙСТВО ====================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🔥 Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"🔥 Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"⚠️  Using CPU (training will be slow)")

print(f"\n{'='*60}")
print(f"Task 1.3: RoBERTa Fine-tuning")
print(f"{'='*60}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
print(f"Batch size: {config['batch_size']}")
print(f"Epochs: {config['epochs']}")
print(f"Max length: {config['max_length']}")
print(f"{'='*60}\n")

# ==================== ЗАГРУЗКА ДАННЫХ ====================
print("📂 Loading data...")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, 'data', 'cleaned_combined_guardian.csv')

df = pd.read_csv(data_path)
print(f"✅ Original dataset size: {len(df)}")

# ==================== ОЧИСТКА ДАННЫХ ====================
print("🧹 Cleaning data...")

df = df.dropna(subset=["cleaned_text", "category"])
df["cleaned_text"] = (
    df["cleaned_text"]
    .astype(str)
    .replace({"nan": "", "None": "", "null": ""})
)
nan_mask = df["cleaned_text"].str.strip().isin(["", "nan", "None", "null"])
df = df[~nan_mask]

print(f"✅ Dataset size after cleaning: {len(df)}")
print(f"📊 Category distribution:")
print(df['category'].value_counts().to_string())

# ==================== ПОДГОТОВКА ====================
texts = df["cleaned_text"].tolist()
categories = df["category"].tolist()

# Кодирование категорий
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(categories)

label_mapping = dict(zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
))
print(f"\n🏷️  Label mapping: {label_mapping}")

# Разделение данных
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,
    random_state=config['random_state']
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=config['random_state']
)

print(f"\n📊 Dataset splits:")
print(f"  Train: {len(train_texts):,} texts")
print(f"  Val:   {len(val_texts):,} texts")
print(f"  Test:  {len(test_texts):,} texts")

# ==================== ТОКЕНИЗАЦИЯ ====================
print("\n🔤 Tokenizing...")

tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config['max_length'],
        padding=False,
        return_attention_mask=True
    )

print("  Tokenizing train...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("  Tokenizing val...")
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("  Tokenizing test...")
tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("✅ Tokenization complete")

# ==================== DATALOADER ====================
print("\n🔄 Setting up DataLoaders...")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

train_dataloader = DataLoader(
    tokenized_train,
    shuffle=True,
    batch_size=config['batch_size'],
    collate_fn=data_collator
)

val_dataloader = DataLoader(
    tokenized_val,
    batch_size=config['batch_size'],
    collate_fn=data_collator
)

test_dataloader = DataLoader(
    tokenized_test,
    batch_size=config['batch_size'],
    collate_fn=data_collator
)

print(f"  Train batches: {len(train_dataloader)}")
print(f"  Val batches: {len(val_dataloader)}")
print(f"  Test batches: {len(test_dataloader)}")

# ==================== МОДЕЛЬ ====================
print(f"\n🤖 Loading {config['model_name']} model...")

model = RobertaForSequenceClassification.from_pretrained(
    config['model_name'],
    num_labels=config['num_labels']
).to(device)

print(f"✅ Model loaded on {device}")
print(f"📊 Total parameters: {model.num_parameters():,}")

# ==================== ОПТИМИЗАТОР ====================
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

total_steps = len(train_dataloader) * config['epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(config['warmup_ratio'] * total_steps),
    num_training_steps=total_steps
)

print(f"\n⚙️  Training setup:")
print(f"  Total steps: {total_steps:,}")
print(f"  Warmup steps: {int(config['warmup_ratio'] * total_steps):,}")

# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="  Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')

    return avg_loss, accuracy, f1_macro, all_predictions, all_labels

# ==================== ОБУЧЕНИЕ ====================
print(f"\n{'='*60}")
print(f"🚀 STARTING TRAINING")
print(f"{'='*60}\n")

best_val_accuracy = 0
train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []

for epoch in range(config['epochs']):
    print(f"\n{'─'*60}")
    print(f"Epoch {epoch + 1}/{config['epochs']}")
    print(f"{'─'*60}")

    # Training
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    print(f"  Train Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_accuracy, val_f1, _, _ = validate(
        model, val_dataloader, device
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(val_f1)

    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Val F1 (macro): {val_f1:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_roberta_model.pth")
        print(f"  ✅ Best model saved! (Val Acc: {val_accuracy:.4f})")

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETED!")
print(f"Best Val Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"{'='*60}")

# ==================== ТЕСТИРОВАНИЕ ====================
print(f"\n📊 Testing best model...")

model.load_state_dict(torch.load("best_roberta_model.pth"))
test_loss, test_accuracy, test_f1, test_predictions, test_labels = validate(
    model, test_dataloader, device
)

print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test F1 (macro): {test_f1:.4f}")
print(f"{'='*60}")

# Classification report
class_names = label_encoder.classes_
print(f"\n📋 Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=class_names))

# ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
print(f"\n💾 Saving results...")

# Создание директорий
results_dir = os.path.join(parent_dir, 'results')
models_dir = os.path.join(parent_dir, 'models')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. Модель и токенизатор
model_dir = os.path.join(models_dir, 'roberta_genre_classifier')
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"  ✅ Model saved to: {model_dir}")

# 2. Label encoder
encoder_path = os.path.join(models_dir, 'roberta_label_encoder.pkl')
joblib.dump(label_encoder, encoder_path)
print(f"  ✅ Label encoder saved to: {encoder_path}")

# 3. Метрики
metrics = {
    "model": "RoBERTa (base)",
    "test_accuracy": float(test_accuracy),
    "test_f1_macro": float(test_f1),
    "best_val_accuracy": float(best_val_accuracy),
    "config": config,
    "device": str(device),
    "train_samples": len(train_texts),
    "val_samples": len(val_texts),
    "test_samples": len(test_texts),
    "total_parameters": model.num_parameters(),
    "training_history": {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": [float(x) for x in val_accuracies],
        "val_f1_scores": [float(x) for x in val_f1_scores]
    }
}

metrics_path = os.path.join(results_dir, 'roberta_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  ✅ Metrics saved to: {metrics_path}")

# 4. Confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
cm_path = os.path.join(results_dir, 'roberta_confusion_matrix.pkl')
with open(cm_path, 'wb') as f:
    pickle.dump({
        'confusion_matrix': cm,
        'class_names': class_names.tolist(),
        'predictions': test_predictions,
        'true_labels': test_labels
    }, f)
print(f"  ✅ Confusion matrix saved to: {cm_path}")

# 5. Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, config['epochs'] + 1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, config['epochs'] + 1), val_losses, 'r-o', label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, config['epochs'] + 1), val_accuracies, 'g-o', label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(range(1, config['epochs'] + 1), val_f1_scores, 'm-o', label='Val F1 (macro)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('F1 Score')
axes[2].set_title('Validation F1 Score (Macro)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
training_curves_path = os.path.join(results_dir, 'roberta_training_curves.png')
plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Training curves saved to: {training_curves_path}")

# 6. Confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('RoBERTa Confusion Matrix')
plt.tight_layout()
cm_plot_path = os.path.join(results_dir, 'roberta_confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Confusion matrix plot saved to: {cm_plot_path}")

# ==================== СРАВНЕНИЕ С ПРЕДЫДУЩИМИ МОДЕЛЯМИ ====================
print(f"\n📊 Comparing with previous models...")

# Загрузка результатов
with open(os.path.join(results_dir, 'tfidf_metrics.json'), 'r') as f:
    tfidf_metrics = json.load(f)

with open(os.path.join(results_dir, 'linguistic_metrics.json'), 'r') as f:
    linguistic_metrics = json.load(f)

comparison = pd.DataFrame({
    'Model': ['TF-IDF + LR', 'Linguistic + RF', 'RoBERTa'],
    'Accuracy': [
        tfidf_metrics['test_accuracy'],
        linguistic_metrics['test_accuracy'],
        test_accuracy
    ],
    'F1 (macro)': [
        tfidf_metrics['test_f1_macro'],
        linguistic_metrics['test_f1_macro'],
        test_f1
    ]
})

print(f"\n{'='*60}")
print(f"BASELINE MODELS COMPARISON")
print(f"{'='*60}")
print(comparison.to_string(index=False))
print(f"{'='*60}")

# График сравнения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(comparison['Model'], comparison['Accuracy'],
           color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Test Accuracy Comparison')
axes[0].set_ylim([0.5, 1.0])
for i, v in enumerate(comparison['Accuracy']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

axes[1].bar(comparison['Model'], comparison['F1 (macro)'],
           color=['#3498db', '#e74c3c', '#2ecc71'])
axes[1].set_ylabel('F1 Score (macro)')
axes[1].set_title('Test F1 Score Comparison')
axes[1].set_ylim([0.5, 1.0])
for i, v in enumerate(comparison['F1 (macro)']):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
comparison_plot_path = os.path.join(results_dir, 'baseline_comparison.png')
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Comparison plot saved to: {comparison_plot_path}")

# ==================== ФИНАЛ ====================
print(f"\n{'='*60}")
print(f"🎉 TASK 1.3 COMPLETED!")
print(f"{'='*60}")
print(f"\nModel: RoBERTa (base)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test F1 (macro): {test_f1:.4f}")

print(f"\n📈 Performance vs expectations:")
print(f"  Expected: ~89%")
print(f"  Achieved: {test_accuracy*100:.2f}%")

if test_accuracy >= 0.89:
    print(f"  ✅ Exceeded expectations!")
elif test_accuracy >= 0.87:
    print(f"  ✅ Meets expectations!")
else:
    print(f"  ⚠️  Below expectations")

print(f"\n⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# Сохранение финального статуса
status = {
    "task": "1.3",
    "model": "RoBERTa",
    "completed": True,
    "test_accuracy": float(test_accuracy),
    "test_f1_macro": float(test_f1),
    "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

status_path = "/tmp/task_13_done.txt"
with open(status_path, 'w') as f:
    f.write(f"🎉 ЗАДАЧА 1.3 ЗАВЕРШЕНА!\n\n")
    f.write(f"Результаты:\n")
    f.write(json.dumps(status, indent=2))

print(f"✅ Status saved to: {status_path}")
print(f"\n🚀 Ready for Task 1.4: Comparative Analysis\n")
