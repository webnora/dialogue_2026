#!/usr/bin/env python3
"""
Phase 4: Statistical Validation

Tasks:
1. McNemar's test for pairwise model comparison
2. Bootstrap confidence intervals for metrics
3. Statistical significance testing
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
import joblib
import json
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
ERROR_ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'error_analysis')

def load_predictions():
    """Load prediction results from error analysis"""
    print("[1/4] Loading prediction data...")

    df = pd.read_csv(os.path.join(ERROR_ANALYSIS_DIR, 'full_predictions.csv'))

    # Get true labels and predictions
    y_true = df['true_category'].values
    tfidf_preds = df['tfidf_pred'].values
    linguistic_preds = df['linguistic_pred'].values
    bert_preds = df['bert_pred'].values

    print(f"  Loaded {len(df)} predictions")
    return y_true, tfidf_preds, linguistic_preds, bert_preds

def mcnemar_test(y_true, y_pred1, y_pred2, model1_name, model2_name):
    """
    Perform McNemar's test for comparing two models.

    H0: The two models have the same error rate
    H1: The two models have different error rates
    """
    print(f"\n  McNemar's test: {model1_name} vs {model2_name}")

    # Contingency table:
    #                    Model2 Correct | Model2 Wrong
    # Model1 Correct |        a        |      b
    # Model1 Wrong   |        c        |      d

    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    # a: both correct
    a = np.sum(correct1 & correct2)

    # b: model1 correct, model2 wrong
    b = np.sum(correct1 & ~correct2)

    # c: model1 wrong, model2 correct
    c = np.sum(~correct1 & correct2)

    # d: both wrong
    d = np.sum(~correct1 & ~correct2)

    print(f"    Contingency table:")
    print(f"                    {model2_name:15s} | {model2_name:15s}")
    print(f"                    {'Correct':15s} | {'Wrong':15s}")
    print(f"    {model1_name:15s} Correct | {a:15d} | {b:15d}")
    print(f"    {model1_name:15s} Wrong   | {c:15d} | {d:15d}")

    # McNemar's test statistic (with continuity correction)
    # chi2 = (|b - c| - 1)^2 / (b + c)

    if b + c == 0:
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"    χ² = {chi2_stat:.4f}")
    print(f"    p-value = {p_value:.4f}")

    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    elif p_value < 0.1:
        significance = "(p < 0.1)"
    else:
        significance = "ns (not significant)"

    print(f"    {significance}")

    return {
        'model1': model1_name,
        'model2': model2_name,
        'a': int(a),
        'b': int(b),
        'c': int(c),
        'd': int(d),
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, random_state=42):
    """
    Compute bootstrap confidence intervals for a metric.

    Returns: (mean, lower_ci, upper_ci)
    """
    np.random.seed(random_state)

    n_samples = len(y_true)
    bootstrapped_scores = []

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        score = metric_fn(y_true_boot, y_pred_boot)
        bootstrapped_scores.append(score)

    bootstrapped_scores = np.array(bootstrapped_scores)

    mean = np.mean(bootstrapped_scores)
    lower_ci = np.percentile(bootstrapped_scores, 2.5)
    upper_ci = np.percentile(bootstrapped_scores, 97.5)

    return mean, lower_ci, upper_ci, bootstrapped_scores

def run_bootstrap_analysis(y_true, tfidf_preds, linguistic_preds, bert_preds, n_bootstrap=1000):
    """Run bootstrap analysis for all models"""
    print(f"\n[2/4] Bootstrap confidence intervals (n={n_bootstrap})...")

    results = {}

    metrics = {
        'accuracy': lambda y_t, y_p: accuracy_score(y_t, y_p),
        'f1_macro': lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
    }

    models = {
        'tfidf': tfidf_preds,
        'linguistic': linguistic_preds,
        'bert': bert_preds
    }

    for model_name, y_pred in models.items():
        print(f"\n  {model_name.upper()}:")
        results[model_name] = {}

        for metric_name, metric_fn in metrics.items():
            mean, lower, upper, _ = bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap)

            print(f"    {metric_name}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

            results[model_name][metric_name] = {
                'mean': float(mean),
                'ci_lower': float(lower),
                'ci_upper': float(upper)
            }

    return results

def compare_overlapping_ci(results, model1, model2, metric='accuracy'):
    """Check if confidence intervals overlap"""
    r1 = results[model1][metric]
    r2 = results[model2][metric]

    # Check for overlap
    if r1['ci_upper'] < r2['ci_lower'] or r2['ci_upper'] < r1['ci_lower']:
        overlap = False
        print(f"    {metric}: {model1} vs {model2} - NO overlap (significant difference)")
    else:
        overlap = True
        print(f"    {metric}: {model1} vs {model2} - overlap (not clearly different)")

    return overlap

def calculate_model_agreement(tfidf_preds, linguistic_preds, bert_preds):
    """Calculate Cohen's Kappa for model agreement"""
    print(f"\n[3/4] Model agreement analysis...")

    # TF-IDF vs BERT
    kappa_tfidf_bert = cohen_kappa_score(tfidf_preds, bert_preds)
    print(f"  TF-IDF vs BERT: κ = {kappa_tfidf_bert:.4f}")

    # TF-IDF vs Linguistic
    kappa_tfidf_ling = cohen_kappa_score(tfidf_preds, linguistic_preds)
    print(f"  TF-IDF vs Linguistic: κ = {kappa_tfidf_ling:.4f}")

    # Linguistic vs BERT
    kappa_ling_bert = cohen_kappa_score(linguistic_preds, bert_preds)
    print(f"  Linguistic vs BERT: κ = {kappa_ling_bert:.4f}")

    return {
        'tfidf_vs_bert': float(kappa_tfidf_bert),
        'tfidf_vs_linguistic': float(kappa_tfidf_ling),
        'linguistic_vs_bert': float(kappa_ling_bert)
    }

def main():
    print("="*80)
    print("Phase 4: Statistical Validation")
    print("="*80)

    # Load predictions
    y_true, tfidf_preds, linguistic_preds, bert_preds = load_predictions()

    # McNemar's tests
    print("\n" + "="*80)
    print("McNemar's Test Results")
    print("="*80)

    mcnemar_results = []

    # Pairwise comparisons
    mcnemar_results.append(mcnemar_test(y_true, tfidf_preds, bert_preds, "TF-IDF", "BERT"))
    mcnemar_results.append(mcnemar_test(y_true, tfidf_preds, linguistic_preds, "TF-IDF", "Linguistic"))
    mcnemar_results.append(mcnemar_test(y_true, linguistic_preds, bert_preds, "Linguistic", "BERT"))

    # Bootstrap analysis
    print("\n" + "="*80)
    print("Bootstrap Confidence Intervals")
    print("="*80)

    bootstrap_results = run_bootstrap_analysis(y_true, tfidf_preds, linguistic_preds, bert_preds, n_bootstrap=1000)

    # CI overlap analysis
    print(f"\n  Confidence Interval Overlap Analysis:")
    compare_overlapping_ci(bootstrap_results, 'tfidf', 'bert', 'accuracy')
    compare_overlapping_ci(bootstrap_results, 'tfidf', 'linguistic', 'accuracy')
    compare_overlapping_ci(bootstrap_results, 'linguistic', 'bert', 'accuracy')

    # Model agreement
    print("\n" + "="*80)
    print("Model Agreement (Cohen's Kappa)")
    print("="*80)

    agreement_results = calculate_model_agreement(tfidf_preds, linguistic_preds, bert_preds)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nKey Findings:")
    print(f"  1. BERT vs TF-IDF: p = {mcnemar_results[0]['p_value']:.4f}")
    print(f"     → {'Significant' if mcnemar_results[0]['significant'] else 'Not significant'} difference")

    print(f"\n  2. TF-IDF accuracy: {bootstrap_results['tfidf']['accuracy']['mean']:.4f} (95% CI: [{bootstrap_results['tfidf']['accuracy']['ci_lower']:.4f}, {bootstrap_results['tfidf']['accuracy']['ci_upper']:.4f}])")
    print(f"     BERT accuracy: {bootstrap_results['bert']['accuracy']['mean']:.4f} (95% CI: [{bootstrap_results['bert']['accuracy']['ci_lower']:.4f}, {bootstrap_results['bert']['accuracy']['ci_upper']:.4f}])")

    print(f"\n  3. Model agreement:")
    print(f"     TF-IDF ↔ BERT: κ = {agreement_results['tfidf_vs_bert']:.4f}")
    print(f"     → {'Substantial agreement' if agreement_results['tfidf_vs_bert'] > 0.6 else 'Moderate agreement'}")

    # Save results
    output_file = os.path.join(RESULTS_DIR, 'statistical_validation.json')

    # Convert numpy types to native Python types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return int(obj)  # Convert bool to int
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(item) for item in obj]
        return obj

    results = to_serializable({
        'mcnemar_tests': mcnemar_results,
        'bootstrap_ci': bootstrap_results,
        'model_agreement': agreement_results,
        'n_samples': len(y_true),
        'n_bootstrap': 1000
    })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    print("\n" + "="*80)
    print("Statistical validation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
