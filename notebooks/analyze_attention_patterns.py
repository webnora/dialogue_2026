#!/usr/bin/env python3
"""
Deep BERT Attention Analysis
Identifies which tokens receive most attention for each genre.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
TEST_DATA_PATH = "data/cleaned_combined_guardian.csv"
MAX_LENGTH = 256
SAMPLES_PER_GENRE = 10  # More samples for better statistics

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n{'='*80}")
print(f"Deep BERT Attention Analysis")
print(f"{'='*80}\n")


def load_samples():
    """Load samples from each genre."""
    print("[1/4] Loading samples...")

    df = pd.read_csv(TEST_DATA_PATH)
    le = joblib.load("models/bert_label_encoder.pkl")
    categories = le.classes_.tolist()

    samples_by_genre = defaultdict(list)

    for category in categories:
        category_df = df[df['category'] == category]
        category_df = category_df[category_df['cleaned_text'].str.len() > 100]

        if len(category_df) >= SAMPLES_PER_GENRE:
            samples = category_df.sample(n=SAMPLES_PER_GENRE, random_state=42)
        else:
            samples = category_df.sample(n=min(5, len(category_df)), random_state=42)

        samples_by_genre[category] = samples['cleaned_text'].tolist()

        print(f"  ✓ {category}: {len(samples)} samples")

    return samples_by_genre, categories


def load_model():
    """Load BERT model with attention."""
    print("\n[2/4] Loading BERT model...")

    model = BertForSequenceClassification.from_pretrained(
        "models/bert_category_classifier",
        output_attentions=True
    ).to(device)

    model.eval()
    tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")

    print("  ✓ Model loaded")
    return model, tokenizer


def extract_top_attention_tokens(model, tokenizer, samples_by_genre, categories):
    """Extract tokens that receive highest attention for each genre."""
    print("\n[3/4] Extracting top attention tokens...")

    genre_top_tokens = defaultdict(Counter)

    for genre in categories:
        print(f"\n  Processing '{genre}'...")

        texts = samples_by_genre[genre]

        for text_idx, text in enumerate(texts):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            ).to(device)

            # Get attention
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # (layers, batch, heads, seq_len, seq_len)

            # Stack: (12, 1, 12, seq_len, seq_len) -> (12, 12, seq_len, seq_len)
            attention_tensor = torch.stack(attentions).squeeze(1)

            # Average across last 6 layers and all heads (typically most informative)
            attention_avg = attention_tensor[6:, :, :, :].mean(dim=(0, 1))  # (seq_len, seq_len)

            # Average attention received by each token (column-wise)
            token_attention = attention_avg.mean(dim=0).cpu().numpy()  # (seq_len,)

            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())

            # Skip special tokens and padding
            valid_indices = [i for i, t in enumerate(tokens)
                           if t not in ['[PAD]', '[CLS]', '[SEP]'] and t.startswith('[') is False]

            # Get top 10 tokens by attention
            valid_attention = [(i, token_attention[i], tokens[i]) for i in valid_indices]
            valid_attention.sort(key=lambda x: x[1], reverse=True)

            # Add to counter (top 10)
            for idx, score, token in valid_attention[:10]:
                # Clean token (remove ## prefix for subwords)
                clean_token = token.replace('##', '')
                genre_top_tokens[genre][clean_token] += score

            if (text_idx + 1) % 5 == 0:
                print(f"    Processed {text_idx + 1}/{len(texts)} samples")

    return genre_top_tokens


def visualize_top_tokens(genre_top_tokens, categories):
    """Visualize top attention tokens for each genre."""
    print("\n[4/4] Creating visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Top 15 Tokens by BERT Attention Weight per Genre',
                 fontsize=16, weight='bold')

    axes = axes.flatten()

    for idx, genre in enumerate(categories):
        ax = axes[idx]

        # Get top 15 tokens for this genre
        top_tokens = genre_top_tokens[genre].most_common(15)

        if not top_tokens:
            ax.text(0.5, 0.5, f'No data for {genre}', ha='center', va='center')
            continue

        tokens = [t[0] for t in top_tokens[::-1]]
        scores = [t[1] for t in top_tokens[::-1]]

        # Normalize scores to percentages
        total = sum(scores)
        scores_pct = [s/total * 100 for s in scores]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tokens)))
        bars = ax.barh(range(len(tokens)), scores_pct, color=colors)

        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=10)
        ax.set_xlabel('Cumulative Attention Weight (%)', fontsize=10, weight='bold')
        ax.set_title(f'{genre}', fontsize=12, weight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(scores_pct) * 1.1)

    # Remove last empty subplot
    axes[-1].axis('off')

    # Add summary text
    summary = (
        "Analysis Method:\n"
        "• Average attention from last 6 layers\n"
        "• " + str(SAMPLES_PER_GENRE) + " samples per genre\n"
        "• Top 15 tokens by cumulative attention\n\n"
        "Higher values indicate tokens\n"
        "BERT focuses on for classification"
    )
    axes[-1].text(0.1, 0.5, summary, fontsize=11, va='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("results/bert_attention_top_tokens.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/bert_attention_top_tokens.png")

    # Print summary
    print("\n" + "="*80)
    print("TOP ATTENTION TOKENS SUMMARY")
    print("="*80)

    for genre in categories:
        print(f"\n{genre.upper()}:")
        top_10 = genre_top_tokens[genre].most_common(10)
        for i, (token, score) in enumerate(top_10, 1):
            print(f"  {i:2d}. {token:15s} (attention: {score:.2f})")


def compare_attention_patterns(genre_top_tokens, categories):
    """Compare attention patterns across genres."""
    print("\n" + "="*80)
    print("CROSS-GENRE COMPARISON")
    print("="*80)

    # Find tokens that are important in multiple genres
    all_tokens = set()
    for genre in categories:
        all_tokens.update([t for t, _ in genre_top_tokens[genre].most_common(20)])

    print(f"\nTokens appearing in top-20 across multiple genres:")

    token_genres = defaultdict(list)
    for token in all_tokens:
        for genre in categories:
            if token in dict(genre_top_tokens[genre]):
                token_genres[token].append(genre)

    # Sort by number of genres
    multi_genre_tokens = [(t, len(gs), gs) for t, gs in token_genres.items() if len(gs) > 1]
    multi_genre_tokens.sort(key=lambda x: x[1], reverse=True)

    for token, num_genres, genres in multi_genre_tokens[:15]:
        print(f"  {token:15s} → {', '.join(genres)}")

    # Genre-specific markers (appear in only one genre's top-20)
    print("\n\nGenre-specific markers (unique to one genre):")
    for genre in categories:
        top_20_tokens = set([t for t, _ in genre_top_tokens[genre].most_common(20)])

        # Remove tokens that appear in other genres
        unique_tokens = top_20_tokens.copy()
        for other_genre in categories:
            if other_genre != genre:
                other_top_20 = set([t for t, _ in genre_top_tokens[other_genre].most_common(20)])
                unique_tokens -= other_top_20

        if unique_tokens:
            print(f"\n{genre}:")
            for token in list(unique_tokens)[:5]:
                score = genre_top_tokens[genre][token]
                print(f"  - {token} ({score:.2f})")


def save_token_analysis(genre_top_tokens, categories):
    """Save detailed token analysis to CSV."""
    print("\n" + "="*80)
    print("Saving token analysis...")

    data = []
    for genre in categories:
        top_20 = genre_top_tokens[genre].most_common(20)
        for rank, (token, score) in enumerate(top_20, 1):
            data.append({
                'genre': genre,
                'rank': rank,
                'token': token,
                'attention_score': score
            })

    df = pd.DataFrame(data)
    df.to_csv("results/bert_attention_tokens.csv", index=False)
    print("  ✓ Saved: results/bert_attention_tokens.csv")


def main():
    # Load data
    samples_by_genre, categories = load_samples()

    # Load model
    model, tokenizer = load_model()

    # Extract attention
    genre_top_tokens = extract_top_attention_tokens(model, tokenizer, samples_by_genre, categories)

    # Visualize
    visualize_top_tokens(genre_top_tokens, categories)

    # Compare
    compare_attention_patterns(genre_top_tokens, categories)

    # Save
    save_token_analysis(genre_top_tokens, categories)

    print("\n" + "="*80)
    print("ATTENTION ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
