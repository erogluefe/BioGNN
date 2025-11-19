#!/usr/bin/env python3
"""
Statistical analysis for biometric evaluation

Includes:
- Cross-validation
- Paired t-test
- Wilcoxon signed-rank test
- Confidence intervals
- Leave-one-subject-out validation
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns


def paired_t_test(
    results1: np.ndarray,
    results2: np.ndarray,
    alpha: float = 0.05
):
    """
    Perform paired t-test to compare two methods

    Args:
        results1: Results from method 1 [n_samples]
        results2: Results from method 2 [n_samples]
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Compute differences
    differences = results1 - results2

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(results1, results2)

    # Effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # Confidence interval
    ci = stats.t.interval(
        1 - alpha,
        len(differences) - 1,
        loc=mean_diff,
        scale=stats.sem(differences)
    )

    results = {
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'cohens_d': cohens_d,
        'confidence_interval': ci
    }

    return results


def wilcoxon_test(
    results1: np.ndarray,
    results2: np.ndarray,
    alpha: float = 0.05
):
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

    Args:
        results1: Results from method 1
        results2: Results from method 2
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(results1, results2)

    results = {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'median_diff': np.median(results1 - results2)
    }

    return results


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
):
    """
    Compute confidence interval for data

    Args:
        data: Data array
        confidence: Confidence level (default 95%)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(
        confidence,
        len(data) - 1,
        loc=mean,
        scale=sem
    )

    return mean, ci[0], ci[1]


def cross_validation_analysis(
    model,
    dataset,
    n_folds: int = 5,
    metric_fn=None
):
    """
    Perform k-fold cross-validation

    Args:
        model: Model to evaluate
        dataset: Dataset
        n_folds: Number of folds
        metric_fn: Function to compute metric

    Returns:
        Dictionary with CV results
    """
    print(f"\nPerforming {n_folds}-fold cross-validation...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{n_folds}")

        # Split data
        # train_data = dataset[train_idx]
        # val_data = dataset[val_idx]

        # Train model on fold
        # ... training code ...

        # Evaluate on validation fold
        # metric = metric_fn(model, val_data)
        # fold_results.append(metric)

        # Placeholder
        fold_results.append(np.random.random())

    fold_results = np.array(fold_results)

    # Compute statistics
    mean_metric = np.mean(fold_results)
    std_metric = np.std(fold_results)
    ci_lower, ci_upper = confidence_interval(fold_results)[1:]

    results = {
        'fold_results': fold_results.tolist(),
        'mean': mean_metric,
        'std': std_metric,
        'confidence_interval_95': (ci_lower, ci_upper),
        'min': np.min(fold_results),
        'max': np.max(fold_results)
    }

    return results


def leave_one_subject_out_validation(
    model,
    dataset,
    subject_ids
):
    """
    Leave-one-subject-out cross-validation

    Tests generalization to unseen subjects

    Args:
        model: Model to evaluate
        dataset: Dataset
        subject_ids: Subject identifiers for each sample

    Returns:
        Dictionary with LOSO results
    """
    print("\nPerforming leave-one-subject-out validation...")

    unique_subjects = np.unique(subject_ids)
    subject_results = []

    for subject in unique_subjects:
        print(f"Testing on subject {subject}")

        # Create train/test split
        test_mask = (subject_ids == subject)
        train_mask = ~test_mask

        # Train on all subjects except this one
        # ... training code ...

        # Test on this subject
        # metric = evaluate(model, dataset[test_mask])
        # subject_results.append(metric)

        # Placeholder
        subject_results.append(np.random.random())

    subject_results = np.array(subject_results)

    # Compute statistics
    results = {
        'subject_results': subject_results.tolist(),
        'mean': np.mean(subject_results),
        'std': np.std(subject_results),
        'median': np.median(subject_results),
        'min': np.min(subject_results),
        'max': np.max(subject_results)
    }

    return results


def plot_comparison(
    method_results: dict,
    metric_name: str = 'EER',
    save_path: str = None
):
    """
    Plot comparison of multiple methods with error bars

    Args:
        method_results: Dictionary mapping method name to results array
        metric_name: Name of metric
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(method_results.keys())
    means = []
    stds = []
    cis = []

    for method in methods:
        results = method_results[method]
        mean, ci_lower, ci_upper = confidence_interval(results)
        means.append(mean)
        stds.append(np.std(results))
        cis.append((ci_lower, ci_upper))

    # Bar plot with error bars
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

    # Add confidence intervals as lines
    for i, (lower, upper) in enumerate(cis):
        ax.plot([i, i], [lower, upper], 'k-', linewidth=2)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(f'{metric_name}', fontsize=12)
    ax.set_title(f'Method Comparison: {metric_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def statistical_comparison(
    method_results: dict,
    baseline_method: str,
    alpha: float = 0.05
):
    """
    Statistical comparison of all methods against baseline

    Args:
        method_results: Dictionary of method results
        baseline_method: Name of baseline method
        alpha: Significance level

    Returns:
        DataFrame with comparison results
    """
    print(f"\nStatistical comparison (baseline: {baseline_method})")

    if baseline_method not in method_results:
        raise ValueError(f"Baseline method '{baseline_method}' not found")

    baseline_results = method_results[baseline_method]

    comparison_results = []

    for method_name, results in method_results.items():
        if method_name == baseline_method:
            continue

        # Paired t-test
        t_test = paired_t_test(baseline_results, results, alpha)

        # Wilcoxon test
        wilcoxon = wilcoxon_test(baseline_results, results, alpha)

        comparison_results.append({
            'method': method_name,
            't_statistic': t_test['t_statistic'],
            't_p_value': t_test['p_value'],
            't_significant': t_test['significant'],
            'wilcoxon_statistic': wilcoxon['statistic'],
            'wilcoxon_p_value': wilcoxon['p_value'],
            'wilcoxon_significant': wilcoxon['significant'],
            'mean_difference': t_test['mean_difference'],
            'cohens_d': t_test['cohens_d']
        })

    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(comparison_results)

    print("\n" + "="*80)
    print(df.to_string())
    print("="*80)

    return df


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis for BioGNN')
    parser.add_argument('--analysis', type=str, default='cv',
                       choices=['cv', 'loso', 'comparison', 'all'],
                       help='Type of statistical analysis')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    args = parser.parse_args()

    print("Running statistical analysis...")

    if args.analysis in ['cv', 'all']:
        cv_results = cross_validation_analysis(
            model=None,
            dataset=None,
            n_folds=args.n_folds
        )
        print("\nCross-validation results:")
        print(f"Mean: {cv_results['mean']:.4f}")
        print(f"Std: {cv_results['std']:.4f}")
        print(f"95% CI: ({cv_results['confidence_interval_95'][0]:.4f}, "
              f"{cv_results['confidence_interval_95'][1]:.4f})")

    if args.analysis in ['loso', 'all']:
        loso_results = leave_one_subject_out_validation(
            model=None,
            dataset=None,
            subject_ids=np.arange(100)  # Placeholder
        )
        print("\nLeave-one-subject-out results:")
        print(f"Mean: {loso_results['mean']:.4f}")
        print(f"Std: {loso_results['std']:.4f}")

    print("\nAnalysis completed!")


if __name__ == '__main__':
    main()
