#!/usr/bin/env python3
"""
Model Performance Comparison Framework

This script provides comprehensive model comparison functionality including
performance metrics, visualizations, and ranking across multiple models.
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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve
)
import mlflow
import mlflow.pytorch

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Framework for comparing multiple models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.comparison_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc',
            'sensitivity', 'specificity', 'mcc'
        ]
        
    def add_model(self, model_name, model_results):
        """Add a model to the comparison."""
        logger.info(f"Adding model: {model_name}")
        self.models[model_name] = model_results
        
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive metrics for a model."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # AUC if probabilities available
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.5
        else:
            metrics['auc'] = 0.5
            
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denom > 0:
            metrics['mcc'] = (tp * tn - fp * fn) / mcc_denom
        else:
            metrics['mcc'] = 0
            
        # Additional metrics
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['total_samples'] = len(y_true)
        
        return metrics
    
    def create_comparison_table(self):
        """Create a comparison table of all models."""
        logger.info("Creating model comparison table...")
        
        comparison_data = []
        
        for model_name, model_data in self.models.items():
            row = {'model': model_name}
            
            # Extract metrics
            if 'metrics' in model_data:
                for metric in self.comparison_metrics:
                    row[metric] = model_data['metrics'].get(metric, 0)
            else:
                # Calculate metrics if not provided
                if all(k in model_data for k in ['y_true', 'y_pred']):
                    metrics = self.calculate_metrics(
                        model_data['y_true'],
                        model_data['y_pred'],
                        model_data.get('y_prob')
                    )
                    for metric in self.comparison_metrics:
                        row[metric] = metrics.get(metric, 0)
                        
            comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Calculate ranks
        for metric in self.comparison_metrics:
            if metric in df.columns:
                # Higher is better for all metrics
                df[f'{metric}_rank'] = df[metric].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [col for col in df.columns if col.endswith('_rank')]
        if rank_cols:
            df['avg_rank'] = df[rank_cols].mean(axis=1)
            df['overall_rank'] = df['avg_rank'].rank()
        
        self.comparison_df = df
        return df
    
    def create_visualizations(self, output_dir):
        """Create comprehensive comparison visualizations."""
        logger.info("Creating comparison visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metric comparison heatmap
        self.create_metric_heatmap(output_dir / "metric_comparison_heatmap.png")
        
        # 2. Radar chart
        self.create_radar_chart(output_dir / "model_comparison_radar.png")
        
        # 3. Bar chart comparison
        self.create_bar_comparison(output_dir / "metric_bar_comparison.png")
        
        # 4. ROC curves comparison
        self.create_roc_comparison(output_dir / "roc_curves_comparison.png")
        
        # 5. Precision-Recall curves
        self.create_pr_comparison(output_dir / "pr_curves_comparison.png")
        
        # 6. Model ranking visualization
        self.create_ranking_plot(output_dir / "model_rankings.png")
        
    def create_metric_heatmap(self, save_path):
        """Create a heatmap of model metrics."""
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
            
        # Select only metric columns
        metric_cols = [col for col in self.comparison_metrics if col in self.comparison_df.columns]
        heatmap_data = self.comparison_df[['model'] + metric_cols].set_index('model')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Score'})
        plt.title('Model Performance Metrics Heatmap')
        plt.xlabel('Model')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_radar_chart(self, save_path):
        """Create a radar chart comparing models."""
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
            
        # Select metrics for radar chart
        radar_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        available_metrics = [m for m in radar_metrics if m in self.comparison_df.columns]
        
        if len(available_metrics) < 3:
            logger.warning("Not enough metrics for radar chart")
            return
            
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.comparison_df)))
        
        for idx, row in self.comparison_df.iterrows():
            values = row[available_metrics].values.tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_bar_comparison(self, save_path):
        """Create grouped bar chart for metric comparison."""
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
            
        # Select key metrics
        bar_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        available_metrics = [m for m in bar_metrics if m in self.comparison_df.columns]
        
        # Prepare data
        n_models = len(self.comparison_df)
        n_metrics = len(available_metrics)
        bar_width = 0.8 / n_models
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars for each model
        for i, (idx, row) in enumerate(self.comparison_df.iterrows()):
            positions = np.arange(n_metrics) + i * bar_width
            values = [row[metric] for metric in available_metrics]
            ax.bar(positions, values, bar_width, label=row['model'], alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(np.arange(n_metrics) + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(available_metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_roc_comparison(self, save_path):
        """Create ROC curve comparison plot."""
        plt.figure(figsize=(8, 8))
        
        for model_name, model_data in self.models.items():
            if 'y_true' in model_data and 'y_prob' in model_data:
                fpr, tpr, _ = roc_curve(model_data['y_true'], model_data['y_prob'])
                auc_score = roc_auc_score(model_data['y_true'], model_data['y_prob'])
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_pr_comparison(self, save_path):
        """Create Precision-Recall curve comparison plot."""
        plt.figure(figsize=(8, 8))
        
        for model_name, model_data in self.models.items():
            if 'y_true' in model_data and 'y_prob' in model_data:
                precision, recall, _ = precision_recall_curve(
                    model_data['y_true'], model_data['y_prob']
                )
                avg_precision = np.mean(precision)
                plt.plot(recall, precision, 
                        label=f'{model_name} (AP = {avg_precision:.3f})', 
                        linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_ranking_plot(self, save_path):
        """Create model ranking visualization."""
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
            
        if 'overall_rank' not in self.comparison_df.columns:
            logger.warning("No ranking data available")
            return
            
        # Sort by overall rank
        ranked_df = self.comparison_df.sort_values('overall_rank')
        
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(ranked_df))
        plt.barh(y_pos, ranked_df['avg_rank'], alpha=0.7)
        
        # Customize
        plt.yticks(y_pos, ranked_df['model'])
        plt.xlabel('Average Rank (lower is better)')
        plt.title('Model Rankings Across All Metrics')
        plt.gca().invert_yaxis()  # Invert y-axis to show best at top
        plt.gca().invert_xaxis()  # Invert x-axis so lower rank is on right
        
        # Add rank labels
        for i, (idx, row) in enumerate(ranked_df.iterrows()):
            plt.text(row['avg_rank'] - 0.1, i, f"#{int(row['overall_rank'])}", 
                    va='center', ha='right', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, output_dir):
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Generate report
        report = {
            'comparison_date': datetime.now().isoformat(),
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'metrics_compared': self.comparison_metrics,
            'rankings': {},
            'best_performers': {},
            'detailed_results': {}
        }
        
        # Find best performer for each metric
        for metric in self.comparison_metrics:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_model = comparison_df.loc[best_idx, 'model']
                best_score = comparison_df.loc[best_idx, metric]
                report['best_performers'][metric] = {
                    'model': best_model,
                    'score': float(best_score)
                }
        
        # Overall rankings
        if 'overall_rank' in comparison_df.columns:
            for idx, row in comparison_df.iterrows():
                report['rankings'][row['model']] = {
                    'overall_rank': int(row['overall_rank']),
                    'average_rank': float(row['avg_rank'])
                }
        
        # Detailed results
        for model_name in self.models:
            model_metrics = comparison_df[comparison_df['model'] == model_name]
            if not model_metrics.empty:
                metrics_dict = model_metrics.iloc[0].to_dict()
                # Remove non-metric columns
                metrics_dict = {k: v for k, v in metrics_dict.items() 
                              if k in self.comparison_metrics and not pd.isna(v)}
                report['detailed_results'][model_name] = metrics_dict
        
        # Save comparison table
        comparison_df.to_csv(output_dir / "model_comparison_table.csv", index=False)
        
        # Save report
        with open(output_dir / "comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log to MLflow
        mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
        with mlflow.start_run(run_name="model_comparison_report"):
            mlflow.log_params({
                'n_models': len(self.models),
                'models_compared': ', '.join(self.models.keys())
            })
            
            # Log best scores
            for metric, info in report['best_performers'].items():
                mlflow.log_metric(f"best_{metric}", info['score'])
                mlflow.set_tag(f"best_{metric}_model", info['model'])
            
            # Log artifacts
            mlflow.log_artifacts(str(output_dir))
        
        return report

def load_mlflow_model_results(experiment_name, run_id=None):
    """Load model results from MLflow."""
    mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment {experiment_name} not found")
        return None
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if run_id:
        run = runs[runs.run_id == run_id]
        if run.empty:
            logger.error(f"Run {run_id} not found")
            return None
    else:
        # Get latest run
        run = runs.iloc[0]
    
    # Extract metrics
    metrics = {}
    metric_columns = [col for col in runs.columns if col.startswith('metrics.')]
    for col in metric_columns:
        metric_name = col.replace('metrics.', '')
        metrics[metric_name] = run[col]
    
    return {
        'run_id': run.run_id,
        'metrics': metrics,
        'params': {col.replace('params.', ''): run[col] 
                  for col in runs.columns if col.startswith('params.')}
    }

def main():
    """Main function for model comparison."""
    logger.info("📊 Starting Model Performance Comparison")
    
    try:
        # Initialize comparator
        comparator = ModelComparator()
        
        # Example: Add models with sample data
        # In practice, you would load actual model results
        
        # Model 1: Baseline
        model1_data = {
            'y_true': np.random.binomial(1, 0.5, 100),
            'y_pred': np.random.binomial(1, 0.5, 100),
            'y_prob': np.random.random(100),
            'model_type': 'DrugBAN_baseline'
        }
        model1_data['metrics'] = comparator.calculate_metrics(
            model1_data['y_true'], 
            model1_data['y_pred'],
            model1_data['y_prob']
        )
        comparator.add_model('DrugBAN_baseline', model1_data)
        
        # Model 2: Optimized
        model2_data = {
            'y_true': model1_data['y_true'],  # Same test set
            'y_pred': np.random.binomial(1, 0.7, 100),  # Better predictions
            'y_prob': np.random.random(100) * 0.3 + 0.5,  # Better calibrated
            'model_type': 'DrugBAN_optimized'
        }
        model2_data['metrics'] = comparator.calculate_metrics(
            model2_data['y_true'], 
            model2_data['y_pred'],
            model2_data['y_prob']
        )
        comparator.add_model('DrugBAN_optimized', model2_data)
        
        # Model 3: Alternative architecture
        model3_data = {
            'y_true': model1_data['y_true'],
            'y_pred': np.random.binomial(1, 0.6, 100),
            'y_prob': np.random.random(100) * 0.4 + 0.3,
            'model_type': 'Alternative_Model'
        }
        model3_data['metrics'] = comparator.calculate_metrics(
            model3_data['y_true'], 
            model3_data['y_pred'],
            model3_data['y_prob']
        )
        comparator.add_model('Alternative_Model', model3_data)
        
        # Generate comparison report
        output_dir = Path("models/comparison")
        report = comparator.generate_comparison_report(output_dir)
        
        print("\n✅ Model Comparison Complete!")
        print("=" * 60)
        print(f"📊 Models Compared: {len(comparator.models)}")
        print("\n🏆 Best Performers:")
        for metric, info in report['best_performers'].items():
            print(f"  {metric}: {info['model']} ({info['score']:.3f})")
        
        print("\n📊 Overall Rankings:")
        if report['rankings']:
            sorted_rankings = sorted(report['rankings'].items(), 
                                   key=lambda x: x[1]['overall_rank'])
            for model, rank_info in sorted_rankings:
                print(f"  #{int(rank_info['overall_rank'])}: {model} "
                     f"(avg rank: {rank_info['average_rank']:.2f})")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print("📊 Visualizations generated:")
        print("  - metric_comparison_heatmap.png")
        print("  - model_comparison_radar.png")
        print("  - metric_bar_comparison.png")
        print("  - roc_curves_comparison.png")
        print("  - pr_curves_comparison.png")
        print("  - model_rankings.png")
        print("📊 Access MLflow UI at: http://127.0.0.1:5000")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())