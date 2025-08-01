#!/usr/bin/env python3
"""
Statistical Significance Testing for Model Comparisons

This script implements various statistical tests for comparing model performance
including paired t-tests, Wilcoxon signed-rank tests, and permutation tests.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalTester:
    """Statistical testing framework for model comparison."""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.test_results = {}
        
    def paired_t_test(self, scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """Perform paired t-test between two models."""
        logger.info(f"Performing paired t-test: {model1_name} vs {model2_name}")
        
        # Calculate differences
        differences = np.array(scores1) - np.array(scores2)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            1 - self.alpha,
            len(differences) - 1,
            loc=np.mean(differences),
            scale=stats.sem(differences)
        )
        
        result = {
            'test': 'paired_t_test',
            'model1': model1_name,
            'model2': model2_name,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'confidence_interval': ci,
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'n_samples': len(scores1)
        }
        
        self.test_results['paired_t_test'] = result
        return result
    
    def wilcoxon_test(self, scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """Perform Wilcoxon signed-rank test (non-parametric alternative)."""
        logger.info(f"Performing Wilcoxon test: {model1_name} vs {model2_name}")
        
        # Perform Wilcoxon test
        w_stat, p_value = stats.wilcoxon(scores1, scores2)
        
        # Calculate effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(1 - p_value/2)
        effect_size_r = z_score / np.sqrt(len(scores1))
        
        # Calculate median difference
        differences = np.array(scores1) - np.array(scores2)
        median_diff = np.median(differences)
        
        result = {
            'test': 'wilcoxon_signed_rank',
            'model1': model1_name,
            'model2': model2_name,
            'median_difference': median_diff,
            'w_statistic': w_stat,
            'p_value': p_value,
            'effect_size_r': effect_size_r,
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'n_samples': len(scores1)
        }
        
        self.test_results['wilcoxon_test'] = result
        return result
    
    def mcnemar_test(self, predictions1, predictions2, labels, model1_name="Model 1", model2_name="Model 2"):
        """Perform McNemar's test for paired binary classifications."""
        logger.info(f"Performing McNemar's test: {model1_name} vs {model2_name}")
        
        # Create contingency table
        # n00: both wrong, n01: model1 wrong & model2 correct
        # n10: model1 correct & model2 wrong, n11: both correct
        n00 = np.sum((predictions1 != labels) & (predictions2 != labels))
        n01 = np.sum((predictions1 != labels) & (predictions2 == labels))
        n10 = np.sum((predictions1 == labels) & (predictions2 != labels))
        n11 = np.sum((predictions1 == labels) & (predictions2 == labels))
        
        contingency_table = np.array([[n00, n01], [n10, n11]])
        
        # Perform McNemar's test
        # Using continuity correction for small samples
        if n01 + n10 > 0:
            chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        else:
            chi2_stat = 0
            p_value = 1.0
        
        result = {
            'test': 'mcnemar',
            'model1': model1_name,
            'model2': model2_name,
            'contingency_table': contingency_table.tolist(),
            'n01_model1_wrong_model2_right': n01,
            'n10_model1_right_model2_wrong': n10,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }
        
        self.test_results['mcnemar_test'] = result
        return result
    
    def bootstrap_test(self, scores1, scores2, n_bootstrap=10000, model1_name="Model 1", model2_name="Model 2"):
        """Perform bootstrap hypothesis test."""
        logger.info(f"Performing bootstrap test with {n_bootstrap} iterations...")
        
        differences = np.array(scores1) - np.array(scores2)
        observed_mean_diff = np.mean(differences)
        
        # Bootstrap under null hypothesis (no difference)
        bootstrap_diffs = []
        n_samples = len(differences)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(differences, size=n_samples, replace=True)
            # Center around 0 (null hypothesis)
            centered_sample = bootstrap_sample - np.mean(bootstrap_sample)
            bootstrap_diffs.append(np.mean(centered_sample))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= abs(observed_mean_diff)),
            np.mean(bootstrap_diffs <= -abs(observed_mean_diff))
        )
        
        # Calculate confidence interval
        ci_lower = np.percentile(differences, (self.alpha/2) * 100)
        ci_upper = np.percentile(differences, (1 - self.alpha/2) * 100)
        
        result = {
            'test': 'bootstrap',
            'model1': model1_name,
            'model2': model2_name,
            'observed_mean_difference': observed_mean_diff,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'n_bootstrap': n_bootstrap,
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }
        
        self.test_results['bootstrap_test'] = result
        return result
    
    def friedman_test(self, all_scores, model_names):
        """Perform Friedman test for comparing multiple models."""
        logger.info(f"Performing Friedman test for {len(model_names)} models...")
        
        # Convert to array (models x folds)
        scores_array = np.array(all_scores)
        
        # Perform Friedman test
        chi2_stat, p_value = stats.friedmanchisquare(*scores_array)
        
        # Calculate average ranks
        ranks = np.array([stats.rankdata(-row) for row in scores_array.T])
        avg_ranks = np.mean(ranks, axis=0)
        
        result = {
            'test': 'friedman',
            'models': model_names,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'average_ranks': dict(zip(model_names, avg_ranks.tolist())),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'n_models': len(model_names),
            'n_folds': scores_array.shape[1]
        }
        
        self.test_results['friedman_test'] = result
        
        # If significant, perform post-hoc Nemenyi test
        if p_value < self.alpha:
            self.nemenyi_post_hoc(scores_array, model_names, avg_ranks)
            
        return result
    
    def nemenyi_post_hoc(self, scores_array, model_names, avg_ranks):
        """Perform Nemenyi post-hoc test after Friedman test."""
        logger.info("Performing Nemenyi post-hoc test...")
        
        n_models = len(model_names)
        n_folds = scores_array.shape[1]
        
        # Critical difference for Nemenyi test
        q_alpha = 2.569  # For alpha=0.05 and 5 models (from statistical tables)
        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_folds))
        
        # Pairwise comparisons
        comparisons = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                significant = rank_diff > cd
                comparisons.append({
                    'model1': model_names[i],
                    'model2': model_names[j],
                    'rank_difference': rank_diff,
                    'critical_difference': cd,
                    'significant': significant
                })
        
        self.test_results['nemenyi_post_hoc'] = {
            'critical_difference': cd,
            'comparisons': comparisons
        }
    
    def create_statistical_report(self, save_path=None):
        """Create comprehensive statistical report with visualizations."""
        logger.info("Creating statistical testing report...")
        
        report = {
            'test_configuration': {
                'alpha': self.alpha,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. P-values summary
        if len(self.test_results) > 0:
            ax = axes[0, 0]
            test_names = []
            p_values = []
            
            for test_name, result in self.test_results.items():
                if 'p_value' in result:
                    test_names.append(test_name.replace('_', ' ').title())
                    p_values.append(result['p_value'])
            
            bars = ax.bar(range(len(test_names)), p_values, alpha=0.7)
            ax.axhline(y=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha='right')
            ax.set_ylabel('P-value')
            ax.set_title('Statistical Test P-values')
            ax.legend()
            ax.set_yscale('log')
            
            # Color bars based on significance
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < self.alpha:
                    bar.set_color('green')
                else:
                    bar.set_color('gray')
        
        # 2. Effect sizes
        ax = axes[0, 1]
        if 'paired_t_test' in self.test_results:
            effect_sizes = {
                "Cohen's d": abs(self.test_results['paired_t_test']['cohens_d']),
            }
            if 'wilcoxon_test' in self.test_results:
                effect_sizes["Wilcoxon r"] = abs(self.test_results['wilcoxon_test']['effect_size_r'])
            
            ax.bar(effect_sizes.keys(), effect_sizes.values(), alpha=0.7, color='skyblue')
            ax.set_ylabel('Effect Size')
            ax.set_title('Effect Sizes')
            ax.axhline(y=0.2, color='orange', linestyle='--', label='Small')
            ax.axhline(y=0.5, color='red', linestyle='--', label='Medium')
            ax.axhline(y=0.8, color='darkred', linestyle='--', label='Large')
            ax.legend()
        
        # 3. Confidence intervals
        ax = axes[1, 0]
        if 'paired_t_test' in self.test_results:
            ci = self.test_results['paired_t_test']['confidence_interval']
            mean_diff = self.test_results['paired_t_test']['mean_difference']
            
            ax.errorbar(0, mean_diff, 
                       yerr=[[mean_diff - ci[0]], [ci[1] - mean_diff]],
                       fmt='o', markersize=10, capsize=10, capthick=2)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([0])
            ax.set_xticklabels(['Mean Difference'])
            ax.set_ylabel('Score Difference')
            ax.set_title(f'{(1-self.alpha)*100}% Confidence Interval')
        
        # 4. Model ranks (if Friedman test performed)
        ax = axes[1, 1]
        if 'friedman_test' in self.test_results:
            ranks_data = self.test_results['friedman_test']['average_ranks']
            models = list(ranks_data.keys())
            ranks = list(ranks_data.values())
            
            bars = ax.bar(range(len(models)), ranks, alpha=0.7, color='coral')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Average Rank')
            ax.set_title('Model Rankings (lower is better)')
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Statistical report visualization saved to: {save_path}")
        
        return report

def compare_models(model1_results, model2_results, model1_name="Model 1", model2_name="Model 2"):
    """Compare two models using multiple statistical tests."""
    logger.info(f"Comparing models: {model1_name} vs {model2_name}")
    
    # Initialize tester
    tester = StatisticalTester(alpha=0.05)
    
    # Extract scores (assuming these are cross-validation results)
    scores1 = model1_results.get('fold_scores', model1_results.get('scores', []))
    scores2 = model2_results.get('fold_scores', model2_results.get('scores', []))
    
    if len(scores1) != len(scores2):
        logger.error("Models have different numbers of scores. Cannot perform paired tests.")
        return None
    
    # Perform tests
    results = {}
    
    # 1. Paired t-test
    t_test_result = tester.paired_t_test(scores1, scores2, model1_name, model2_name)
    results['paired_t_test'] = t_test_result
    
    # 2. Wilcoxon test
    wilcoxon_result = tester.wilcoxon_test(scores1, scores2, model1_name, model2_name)
    results['wilcoxon_test'] = wilcoxon_result
    
    # 3. Bootstrap test
    bootstrap_result = tester.bootstrap_test(scores1, scores2, n_bootstrap=10000, 
                                           model1_name=model1_name, model2_name=model2_name)
    results['bootstrap_test'] = bootstrap_result
    
    # 4. McNemar test (if predictions available)
    if 'predictions' in model1_results and 'predictions' in model2_results:
        preds1 = model1_results['predictions']
        preds2 = model2_results['predictions']
        labels = model1_results.get('labels', model2_results.get('labels', []))
        
        if len(preds1) == len(preds2) == len(labels):
            mcnemar_result = tester.mcnemar_test(preds1, preds2, labels, model1_name, model2_name)
            results['mcnemar_test'] = mcnemar_result
    
    # Create report
    report = tester.create_statistical_report("models/statistical_tests/comparison_report.png")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"statistical_comparison_{model1_name}_vs_{model2_name}"):
        mlflow.log_params({
            'model1': model1_name,
            'model2': model2_name,
            'alpha': tester.alpha,
            'n_tests': len(results)
        })
        
        for test_name, result in results.items():
            if 'p_value' in result:
                mlflow.log_metric(f"{test_name}_p_value", result['p_value'])
                mlflow.log_metric(f"{test_name}_significant", int(result['significant']))
        
        mlflow.log_artifact("models/statistical_tests/comparison_report.png")
    
    return results

def compare_multiple_models(all_model_results, model_names):
    """Compare multiple models using Friedman test."""
    logger.info(f"Comparing {len(model_names)} models...")
    
    # Initialize tester
    tester = StatisticalTester(alpha=0.05)
    
    # Extract scores for all models
    all_scores = []
    for model_results in all_model_results:
        scores = model_results.get('fold_scores', model_results.get('scores', []))
        all_scores.append(scores)
    
    # Perform Friedman test
    friedman_result = tester.friedman_test(all_scores, model_names)
    
    # Create report
    report = tester.create_statistical_report("models/statistical_tests/multi_model_comparison.png")
    
    # Log to MLflow
    with mlflow.start_run(run_name="statistical_comparison_multiple_models"):
        mlflow.log_params({
            'n_models': len(model_names),
            'models': ', '.join(model_names),
            'alpha': tester.alpha
        })
        
        mlflow.log_metric("friedman_chi2", friedman_result['chi2_statistic'])
        mlflow.log_metric("friedman_p_value", friedman_result['p_value'])
        mlflow.log_metric("friedman_significant", int(friedman_result['significant']))
        
        for model, rank in friedman_result['average_ranks'].items():
            mlflow.log_metric(f"rank_{model}", rank)
        
        mlflow.log_artifact("models/statistical_tests/multi_model_comparison.png")
    
    return report

def main():
    """Main function for statistical testing."""
    logger.info("🔬 Starting Statistical Significance Testing")
    
    try:
        # Create output directory
        output_dir = Path("models/statistical_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Example: Load cross-validation results
        # In practice, you would load actual model results
        example_model1_results = {
            'fold_scores': [0.85, 0.87, 0.83, 0.86, 0.84],
            'model_name': 'DrugBAN_base'
        }
        
        example_model2_results = {
            'fold_scores': [0.88, 0.89, 0.87, 0.88, 0.87],
            'model_name': 'DrugBAN_optimized'
        }
        
        # Compare two models
        comparison_results = compare_models(
            example_model1_results,
            example_model2_results,
            model1_name="DrugBAN_base",
            model2_name="DrugBAN_optimized"
        )
        
        # Save results
        with open(output_dir / "statistical_test_results.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print("\n✅ Statistical Testing Complete!")
        print("=" * 60)
        
        # Print summary
        for test_name, result in comparison_results.items():
            if 'p_value' in result:
                print(f"\n{test_name.replace('_', ' ').title()}:")
                print(f"  P-value: {result['p_value']:.4f}")
                print(f"  Significant: {'Yes' if result['significant'] else 'No'}")
                
                if test_name == 'paired_t_test':
                    print(f"  Mean difference: {result['mean_difference']:.4f}")
                    print(f"  Cohen's d: {result['cohens_d']:.3f}")
                elif test_name == 'wilcoxon_test':
                    print(f"  Median difference: {result['median_difference']:.4f}")
                    print(f"  Effect size r: {result['effect_size_r']:.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("📊 Visualization: comparison_report.png")
        print("📊 Access MLflow UI at: http://127.0.0.1:5000")
        
        return 0
        
    except Exception as e:
        logger.error(f"Statistical testing failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())