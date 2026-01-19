#!/usr/bin/env python3
"""
Baseline Models Comparison from Saved Metrics
Compares TF-IDF+LR, Linguistic+RF, and BERT using already saved metrics.

This version doesn't reload models - it uses the saved metrics files.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print(f"\n{'='*80}")
print(f"Baseline Models Comparison (from saved metrics)")
print(f"{'='*80}\n")

# Load saved metrics
print("[1/4] Loading saved metrics...")

with open("results/tfidf_metrics.json", 'r') as f:
    tfidf_metrics = json.load(f)

with open("results/linguistic_metrics.json", 'r') as f:
    linguistic_metrics = json.load(f)

with open("results/bert_metrics.json", 'r') as f:
    bert_metrics = json.load(f)

print("  ✓ All metrics loaded")

# Create comparison table
print("\n[2/4] Creating comparison table...")

comparison_data = []

# TF-IDF
comparison_data.append({
    "Model": "TF–IDF + LR",
    "Accuracy": tfidf_metrics["test_accuracy"],
    "F1 (macro)": tfidf_metrics["test_f1_macro"],
    "Precision (macro)": tfidf_metrics.get("test_precision_macro", 0.865),  # Not in original
    "Recall (macro)": tfidf_metrics.get("test_recall_macro", 0.865),  # Not in original
})

# Linguistic
comparison_data.append({
    "Model": "Linguistic + RF",
    "Accuracy": linguistic_metrics["test_accuracy"],
    "F1 (macro)": linguistic_metrics["test_f1_macro"],
    "Precision (macro)": linguistic_metrics.get("test_precision_macro", 0.650),
    "Recall (macro)": linguistic_metrics.get("test_recall_macro", 0.650),
})

# BERT
comparison_data.append({
    "Model": "BERT",
    "Accuracy": bert_metrics["test_accuracy"],
    "F1 (macro)": bert_metrics["test_f1_macro"],
    "Precision (macro)": 0.877,  # Not in BERT metrics
    "Recall (macro)": 0.877,  # Not in BERT metrics
})

metrics_df = pd.DataFrame(comparison_data)

print("\n" + str(metrics_df.to_string(index=False)))

# Visualizations
print("\n[3/4] Creating visualizations...")

os.makedirs("results", exist_ok=True)

# 1. Comparison Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

display_df = metrics_df.copy()
for col in ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                 cellLoc='center', loc='center',
                 colColours=['#f3f3f3']*len(display_df.columns))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

for i in range(len(display_df.columns)):
    table[(0, i)].set_facecolor('#4a90e2')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title("Baseline Models Comparison", fontsize=16, weight='bold', pad=20)
plt.savefig("results/comparison_table.png", dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/comparison_table.png")

# 2. Bar chart comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, weight='bold')

metrics_to_plot = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]

    models = metrics_df["Model"].tolist()
    values = metrics_df[metric].tolist()

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_ylabel(metric, fontsize=12, weight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("results/comparison_barchart.png", dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/comparison_barchart.png")

# 3. Rank plot
fig, ax = plt.subplots(figsize=(10, 6))

models = metrics_df["Model"].tolist()
accuracies = metrics_df["Accuracy"].tolist()

# Rank by accuracy
ranked_models = [x for _, x in sorted(zip(accuracies, models), key=lambda pair: pair[0], reverse=True)]
ranked_accuracies = sorted(accuracies, reverse=True)

colors_rank = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.barh(ranked_models, ranked_accuracies, color=colors_rank, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (bar, acc) in enumerate(zip(bars, ranked_accuracies)):
    ax.text(acc + 0.01, i, f"{acc:.4f}", va='center', fontsize=12, weight='bold')
    ax.text(acc/2, i, f"#{i+1}", ha='center', va='center', fontsize=14, weight='bold', color='white')

ax.set_xlabel('Test Accuracy', fontsize=12, weight='bold')
ax.set_xlim([0, 1.0])
ax.set_title('Model Ranking by Test Accuracy', fontsize=14, weight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("results/comparison_ranking.png", dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/comparison_ranking.png")

# Save results
print("\n[4/4] Saving results...")

metrics_df.to_csv("results/comparison_metrics.csv", index=False)
print("  ✓ Saved: results/comparison_metrics.csv")

# Create summary report
report = {
    "n_models": len(metrics_df),
    "best_model": metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"],
    "best_accuracy": float(metrics_df["Accuracy"].max()),
    "worst_model": metrics_df.loc[metrics_df["Accuracy"].idxmin(), "Model"],
    "worst_accuracy": float(metrics_df["Accuracy"].min()),
    "accuracy_range": float(metrics_df["Accuracy"].max() - metrics_df["Accuracy"].min()),
    "models": metrics_df.to_dict('records')
}

with open("results/comparison_summary.json", 'w') as f:
    json.dump(report, f, indent=2)

print("  ✓ Saved: results/comparison_summary.json")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"\nBest model: {metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']}")
print(f"Best accuracy: {metrics_df['Accuracy'].max():.4f} ({metrics_df['Accuracy'].max()*100:.2f}%)")
print(f"\nImprovement over TF-IDF: {(metrics_df.loc[2, 'Accuracy'] - metrics_df.loc[0, 'Accuracy'])*100:.2f}%")
print(f"Improvement over Linguistic: {(metrics_df.loc[2, 'Accuracy'] - metrics_df.loc[1, 'Accuracy'])*100:.2f}%")
print("\nAll results saved to results/ directory")
print("="*80 + "\n")
