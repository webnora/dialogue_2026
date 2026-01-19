#!/usr/bin/env python3
"""
BERT Attention Visualization for Genre Classification
Analyzes what BERT focuses on when classifying different genres.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
TEST_DATA_PATH = "data/cleaned_combined_guardian.csv"
MAX_LENGTH = 256
BATCH_SIZE = 4
SAMPLES_PER_GENRE = 3  # Number of examples to visualize per genre

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n{'='*80}")
print(f"BERT Attention Visualization for Genre Classification")
print(f"{'='*80}\n")
print(f"Using device: {device}\n")


def load_test_samples():
    """Load test samples for each genre."""
    print("[1/6] Loading test data and sampling examples...")

    # Load full dataset
    df = pd.read_csv(TEST_DATA_PATH)

    # Load label encoder to get categories
    le = joblib.load("models/bert_label_encoder.pkl")
    categories = le.classes_.tolist()

    print(f"  Categories: {categories}")

    # Sample texts from each genre
    sampled_texts = []
    sampled_labels = []
    sampled_true_labels = []

    for category in categories:
        category_df = df[df['category'] == category]

        # Filter by minimum length (avoid too short texts)
        category_df = category_df[category_df['cleaned_text'].str.len() > 100]

        if len(category_df) >= SAMPLES_PER_GENRE:
            samples = category_df.sample(n=SAMPLES_PER_GENRE, random_state=42)
        else:
            samples = category_df.sample(n=min(3, len(category_df)), random_state=42)

        sampled_texts.extend(samples['cleaned_text'].tolist())
        sampled_labels.extend([category] * len(samples))
        sampled_true_labels.extend([le.transform([category])[0]] * len(samples))

        print(f"  ✓ Sampled {len(samples)} texts from '{category}'")

    print(f"\n  Total samples: {len(sampled_texts)}")
    return sampled_texts, sampled_labels, sampled_true_labels, categories


def load_model_with_attention():
    """Load BERT model with attention output enabled."""
    print("\n[2/6] Loading BERT model...")

    model = BertForSequenceClassification.from_pretrained(
        "models/bert_category_classifier",
        output_attentions=True  # CRITICAL: enable attention output
    ).to(device)

    tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")

    print("  ✓ Model loaded with attention output enabled")
    return model, tokenizer


def tokenize_texts(texts, tokenizer):
    """Tokenize texts for BERT."""
    print("\n[3/6] Tokenizing texts...")

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

    test_dataset = Dataset.from_dict({"text": texts})
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"  ✓ Tokenized {len(texts)} texts")
    return tokenized_test


def extract_attention_weights(model, tokenized_dataset, true_labels):
    """Extract attention weights and predictions for all samples."""
    print("\n[4/6] Extracting attention weights...")

    model.eval()

    all_attentions = []
    all_tokens = []
    all_predictions = []
    all_true_labels = []

    for idx in range(len(tokenized_dataset)):
        # Get single example
        input_ids = torch.tensor([tokenized_dataset[idx]['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([tokenized_dataset[idx]['attention_mask']], dtype=torch.long).to(device)

        # Get tokens
        tokens = model.config.id2token.get if hasattr(model.config, 'id2token') else None
        if not tokens:
            tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # Forward pass with attention
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            attentions = outputs.attentions  # Tuple of (layers, batch, heads, seq_len, seq_len)

        # Get prediction
        pred = torch.argmax(logits, dim=1).cpu().item()

        # Stack attentions: (num_layers, batch, num_heads, seq_len, seq_len)
        attention_tensor = torch.stack(attentions)  # (12, 1, 12, seq_len, seq_len)
        attention_tensor = attention_tensor.squeeze(1)  # (12, 12, seq_len, seq_len)

        # Store
        all_attentions.append(attention_tensor.cpu().numpy())
        all_tokens.append(tokens)
        all_predictions.append(pred)
        all_true_labels.append(true_labels[idx])

        print(f"  ✓ Processed sample {idx+1}/{len(tokenized_dataset)} (True: {all_true_labels[-1]}, Pred: {all_predictions[-1]})")

    return all_attentions, all_tokens, all_predictions, all_true_labels


def visualize_sample_attention(tokens, attention, layer, head, save_path):
    """Visualize attention from a specific layer and head."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get attention for this layer and head
    attn_matrix = attention[layer, head]  # (seq_len, seq_len)

    # Only show first 100 tokens to avoid overcrowding
    max_tokens = min(100, len(tokens))
    attn_matrix = attn_matrix[:max_tokens, :max_tokens]
    tokens = tokens[:max_tokens]

    # Create heatmap
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_title(f'Attention Heatmap - Layer {layer+1}, Head {head+1}', fontsize=14, weight='bold')
    ax.set_xlabel('Keys', fontsize=12, weight='bold')
    ax.set_ylabel('Queries', fontsize=12, weight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_average_genre_attention(all_attentions, all_tokens, all_labels, categories, save_dir):
    """Visualize average attention patterns per genre."""
    print("\n[5/6] Creating average attention visualizations per genre...")

    import os
    os.makedirs(save_dir, exist_ok=True)

    # Group by genre
    genre_attentions = defaultdict(list)
    genre_tokens = defaultdict(list)

    for attn, tokens, label in zip(all_attentions, all_tokens, all_labels):
        genre_name = categories[label]
        genre_attentions[genre_name].append(attn)
        genre_tokens[genre_name].append(tokens)

    # For each genre, average attention and visualize
    for genre in categories:
        if genre not in genre_attentions:
            continue

        print(f"  Processing '{genre}'...")

        # Average attention across all samples of this genre
        avg_attention = np.mean(genre_attentions[genre], axis=0)  # (12, 12, seq_len, seq_len)

        # Use first sample's tokens for display
        sample_tokens = genre_tokens[genre][0]
        max_tokens = min(50, len(sample_tokens))

        # Visualize specific layers/heads that are typically important
        important_heads = [(0, 0), (5, 0), (10, 0), (11, 0)]  # Layer, Head

        for layer, head in important_heads:
            if layer < avg_attention.shape[0]:
                fig, ax = plt.subplots(figsize=(10, 8))

                attn_matrix = avg_attention[layer, head, :max_tokens, :max_tokens]

                im = ax.imshow(attn_matrix, cmap='RdYlBu_r', aspect='auto')

                ax.set_xticks(range(len(sample_tokens[:max_tokens])))
                ax.set_yticks(range(len(sample_tokens[:max_tokens])))
                ax.set_xticklabels(sample_tokens[:max_tokens], rotation=90, fontsize=8)
                ax.set_yticklabels(sample_tokens[:max_tokens], fontsize=8)

                ax.set_title(f'{genre} - Avg Attention (Layer {layer+1}, Head {head+1})',
                            fontsize=14, weight='bold')
                ax.set_xlabel('Keys', fontsize=12)
                ax.set_ylabel('Queries', fontsize=12)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Attention Weight', rotation=270, labelpad=15)

                plt.tight_layout()
                plt.savefig(f"{save_dir}/{genre}_layer{layer+1}_head{head+1}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()

        print(f"    ✓ Saved {len(important_heads)} visualizations for '{genre}'")

    print(f"  ✓ All genre visualizations saved to {save_dir}/")


def analyze_attention_statistics(all_attentions, all_tokens, all_labels, all_predictions, categories):
    """Analyze attention patterns and save statistics."""
    print("\n[6/6] Analyzing attention patterns...")

    stats = {
        'num_samples': len(all_attentions),
        'num_layers': all_attentions[0].shape[0],
        'num_heads': all_attentions[0].shape[1],
        'samples_by_genre': {},
        'attention_stats': {}
    }

    # Count samples per genre
    for label in all_labels:
        genre = categories[label]
        stats['samples_by_genre'][genre] = stats['samples_by_genre'].get(genre, 0) + 1

    # Compute average attention statistics
    layer_averages = []
    head_averages = []

    for attn in all_attentions:
        # Average over sequence length
        avg_attn = attn.mean(axis=(2, 3))  # (num_layers, num_heads)

        layer_averages.append(avg_attn.mean(axis=1))  # (num_layers,)
        head_averages.append(avg_attn.mean(axis=0))  # (num_heads,)

    stats['attention_stats']['mean_attention_by_layer'] = np.mean(layer_averages, axis=0).tolist()
    stats['attention_stats']['mean_attention_by_head'] = np.mean(head_averages, axis=0).tolist()

    # Find which layers/heads are most active
    stats['attention_stats']['most_attentive_layer'] = int(np.argmax(stats['attention_stats']['mean_attention_by_layer']))
    stats['attention_stats']['most_attentive_head'] = int(np.argmax(stats['attention_stats']['mean_attention_by_head']))

    # Print summary
    print(f"\n  Samples analyzed: {stats['num_samples']}")
    print(f"\n  Samples by genre:")
    for genre, count in stats['samples_by_genre'].items():
        print(f"    {genre}: {count}")

    print(f"\n  Most attentive layer: {stats['attention_stats']['most_attentive_layer'] + 1}")
    print(f"  Most attentive head: {stats['attention_stats']['most_attentive_head'] + 1}")

    # Layer-wise analysis
    print(f"\n  Mean attention by layer:")
    for i, val in enumerate(stats['attention_stats']['mean_attention_by_layer']):
        print(f"    Layer {i+1}: {val:.6f}")

    # Save statistics
    import json
    with open("results/bert_attention_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  ✓ Saved: results/bert_attention_stats.json")

    return stats


def visualize_layer_wise_attention(all_attentions, all_tokens, sample_idx, save_path):
    """Visualize how attention evolves across layers for a single sample."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Attention Evolution Across Layers - Sample {sample_idx}',
                 fontsize=16, weight='bold')

    attention = all_attentions[sample_idx]
    tokens = all_tokens[sample_idx]

    # Use head 0 for visualization
    head = 0
    max_tokens = min(50, len(tokens))

    for layer in range(12):
        ax = axes[layer // 4, layer % 4]

        attn_matrix = attention[layer, head, :max_tokens, :max_tokens]

        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer+1}', fontsize=10, weight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load data
    texts, labels, true_labels, categories = load_test_samples()

    # Load model
    model, tokenizer = load_model_with_attention()

    # Tokenize
    tokenized_dataset = tokenize_texts(texts, tokenizer)

    # Extract attention
    all_attentions, all_tokens, all_predictions, all_true_labels = extract_attention_weights(
        model, tokenized_dataset, true_labels
    )

    # Visualize individual samples (first 2)
    print("\n" + "="*80)
    print("Creating individual sample visualizations...")
    print("="*80)

    import os
    os.makedirs("results/attention_individual", exist_ok=True)

    for i in range(min(2, len(all_tokens))):
        # Layer 11 (last layer), Head 0
        visualize_sample_attention(
            all_tokens[i],
            all_attentions[i],
            layer=11,
            head=0,
            save_path=f"results/attention_individual/sample_{i}_layer11_head0.png"
        )

        # Layer-wise evolution
        visualize_layer_wise_attention(
            all_attentions,
            all_tokens,
            i,
            save_path=f"results/attention_individual/sample_{i}_layer_evolution.png"
        )

        print(f"  ✓ Saved visualizations for sample {i+1}")

    # Average attention per genre
    print("\n" + "="*80)
    print("Creating genre-level attention visualizations...")
    print("="*80)

    visualize_average_genre_attention(
        all_attentions,
        all_tokens,
        all_true_labels,
        categories,
        "results/attention_by_genre"
    )

    # Statistics
    print("\n" + "="*80)
    print("Analyzing attention statistics...")
    print("="*80)

    stats = analyze_attention_statistics(
        all_attentions,
        all_tokens,
        all_true_labels,
        all_predictions,
        categories
    )

    print("\n" + "="*80)
    print("BERT ATTENTION VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - results/attention_individual/ (individual samples)")
    print(f"  - results/attention_by_genre/ (average per genre)")
    print(f"  - results/bert_attention_stats.json (statistics)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
