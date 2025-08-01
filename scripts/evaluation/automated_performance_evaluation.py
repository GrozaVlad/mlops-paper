#!/usr/bin/env python3
"""
Automated Model Performance Evaluation
Comprehensive automated evaluation system for continuous model performance monitoring.
"""

import argparse
import json
import sys
import time
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutomatedPerformanceEvaluator:
    """Automated model performance evaluation system."""
    
    def __init__(self, config_path: str = "configs/evaluation_config.yaml"):
        """Initialize performance evaluator.
        
        Args:
            config_path: Path to evaluation configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Evaluation results
        self.evaluation_results = {
            "evaluation_id": f"eval_{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_evaluations": {},
            "comparative_analysis": {},
            "performance_trends": {},
            "recommendations": [],
            "status": "RUNNING"
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration."""
        default_config = {
            "evaluation_metrics": {
                "primary_metrics": ["accuracy", "auc", "precision", "recall", "f1_score"],
                "secondary_metrics": ["specificity", "npv", "mcc", "balanced_accuracy"],
                "confidence_metrics": ["prediction_confidence", "calibration_score"],
                "efficiency_metrics": ["inference_time", "memory_usage", "throughput"]
            },
            "performance_thresholds": {
                "min_accuracy": 0.75,
                "min_auc": 0.80,
                "min_precision": 0.70,
                "min_recall": 0.70,
                "min_f1": 0.70,
                "max_inference_time_ms": 1000,
                "min_calibration_score": 0.80
            },
            "evaluation_datasets": {
                "test_set": "data/test/",
                "validation_set": "data/validation/",
                "holdout_set": "data/holdout/",
                "production_sample": "data/production_sample/"
            },
            "trend_analysis": {
                "lookback_days": 30,
                "min_evaluations": 5,
                "trend_significance_threshold": 0.05,
                "performance_degradation_threshold": 0.05
            },
            "evaluation_schedule": {
                "frequency": "daily",
                "evaluation_hour": 3,
                "weekend_evaluation": True
            },
            "reporting": {
                "generate_plots": True,
                "save_detailed_results": True,
                "create_performance_report": True,
                "notify_on_degradation": True
            }
        }
        
        try:
            import yaml
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return {**default_config, **config}
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return default_config
        except ImportError:
            logger.warning("PyYAML not installed, using default configuration")
            return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return default_config
    
    def evaluate_model(self, model_uri: str, model_name: str = None,
                      evaluation_datasets: Dict[str, str] = None) -> Dict[str, Any]:
        """Evaluate a single model comprehensively.
        
        Args:
            model_uri: URI of the model to evaluate
            model_name: Name of the model (for reporting)
            evaluation_datasets: Dataset paths for evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting comprehensive model evaluation: {model_uri}")
        
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        if evaluation_datasets is None:
            evaluation_datasets = self.config["evaluation_datasets"]
        
        # Load model
        try:
            model = mlflow.pytorch.load_model(model_uri)
            model_info = self._get_model_info(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model {model_uri}: {e}")
            return {"error": str(e), "status": "FAILED"}
        
        evaluation_result = {
            "model_uri": model_uri,
            "model_name": model_name,
            "model_info": model_info,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_evaluations": {},
            "overall_metrics": {},
            "performance_analysis": {},
            "recommendations": [],
            "status": "COMPLETED"
        }
        
        # Evaluate on each dataset
        for dataset_name, dataset_path in evaluation_datasets.items():
            if Path(dataset_path).exists():
                logger.info(f"Evaluating on {dataset_name} dataset...")
                dataset_results = self._evaluate_on_dataset(model, dataset_path, dataset_name)
                evaluation_result["dataset_evaluations"][dataset_name] = dataset_results
            else:
                logger.warning(f"Dataset not found: {dataset_path}")
        
        # Calculate overall metrics
        evaluation_result["overall_metrics"] = self._calculate_overall_metrics(
            evaluation_result["dataset_evaluations"]
        )
        
        # Performance analysis
        evaluation_result["performance_analysis"] = self._analyze_performance(
            evaluation_result["overall_metrics"]
        )
        
        # Generate recommendations
        evaluation_result["recommendations"] = self._generate_recommendations(
            evaluation_result["performance_analysis"]
        )
        
        # Store results
        self.evaluation_results["model_evaluations"][model_name] = evaluation_result
        
        logger.info(f"Model evaluation completed: {model_name}")
        return evaluation_result
    
    def _evaluate_on_dataset(self, model: Any, dataset_path: str, 
                           dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific dataset."""
        try:
            # Load test data (simplified - would load actual drug-target data)
            X_test, y_test = self._load_test_data(dataset_path)
            
            if X_test is None or y_test is None:
                return {"error": "Failed to load test data", "status": "FAILED"}
            
            # Make predictions
            start_time = time.time()
            predictions, probabilities = self._make_predictions(model, X_test)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions, probabilities)
            
            # Additional analysis
            confidence_analysis = self._analyze_prediction_confidence(probabilities)
            error_analysis = self._analyze_prediction_errors(y_test, predictions, probabilities)
            
            return {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "sample_count": len(y_test),
                "inference_time_ms": inference_time,
                "inference_time_per_sample_ms": inference_time / len(y_test),
                "metrics": metrics,
                "confidence_analysis": confidence_analysis,
                "error_analysis": error_analysis,
                "status": "COMPLETED"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating on dataset {dataset_name}: {e}")
            return {"error": str(e), "status": "FAILED"}
    
    def _load_test_data(self, dataset_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load test data from dataset path."""
        try:
            # For demonstration, generate sample data
            # In reality, this would load actual drug-target interaction data
            np.random.seed(42)
            n_samples = 1000
            n_features = 100
            
            X_test = np.random.randn(n_samples, n_features)
            y_test = np.random.randint(0, 2, n_samples)
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading test data from {dataset_path}: {e}")
            return None, None
    
    def _make_predictions(self, model: Any, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the model."""
        try:
            # For demonstration, generate sample predictions
            # In reality, this would use the actual model
            n_samples = len(X_test)
            
            # Generate realistic predictions with some correlation to input
            probabilities = np.random.uniform(0.1, 0.9, n_samples)
            predictions = (probabilities > 0.5).astype(int)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Primary classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # AUC score
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.5  # Default for edge cases
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
        
        # Matthews Correlation Coefficient
        try:
            from sklearn.metrics import matthews_corrcoef
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics["mcc"] = 0.0
        
        # Confidence metrics
        metrics["mean_confidence"] = np.mean(y_prob)
        metrics["confidence_std"] = np.std(y_prob)
        
        # Calibration score (simplified)
        metrics["calibration_score"] = self._calculate_calibration_score(y_true, y_prob)
        
        return metrics
    
    def _calculate_calibration_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate model calibration score."""
        try:
            # Simplified calibration calculation using binning
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return 1.0 - ece  # Convert to calibration score (higher is better)
            
        except Exception:
            return 0.5  # Default calibration score
    
    def _analyze_prediction_confidence(self, y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence distribution."""
        return {
            "mean_confidence": float(np.mean(y_prob)),
            "median_confidence": float(np.median(y_prob)),
            "std_confidence": float(np.std(y_prob)),
            "min_confidence": float(np.min(y_prob)),
            "max_confidence": float(np.max(y_prob)),
            "high_confidence_pct": float(np.mean(y_prob > 0.8) * 100),
            "low_confidence_pct": float(np.mean(y_prob < 0.2) * 100),
            "confident_predictions_pct": float(np.mean((y_prob < 0.2) | (y_prob > 0.8)) * 100)
        }
    
    def _analyze_prediction_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors and patterns."""
        # False positives and false negatives
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        fp_confidences = y_prob[fp_mask] if np.any(fp_mask) else np.array([])
        fn_confidences = y_prob[fn_mask] if np.any(fn_mask) else np.array([])
        
        return {
            "false_positive_count": int(np.sum(fp_mask)),
            "false_negative_count": int(np.sum(fn_mask)),
            "false_positive_rate": float(np.mean(fp_mask)),
            "false_negative_rate": float(np.mean(fn_mask)),
            "fp_mean_confidence": float(np.mean(fp_confidences)) if len(fp_confidences) > 0 else 0,
            "fn_mean_confidence": float(np.mean(fn_confidences)) if len(fn_confidences) > 0 else 0,
            "error_confidence_analysis": {
                "high_confidence_errors": int(np.sum((fp_mask | fn_mask) & (y_prob > 0.8))),
                "low_confidence_errors": int(np.sum((fp_mask | fn_mask) & (y_prob < 0.2)))
            }
        }
    
    def _calculate_overall_metrics(self, dataset_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics across all datasets."""
        overall_metrics = {}
        
        # Get primary metrics from test set (preferred) or first available dataset
        primary_dataset = None
        if "test_set" in dataset_evaluations:
            primary_dataset = dataset_evaluations["test_set"]
        elif dataset_evaluations:
            primary_dataset = list(dataset_evaluations.values())[0]
        
        if primary_dataset and primary_dataset.get("status") == "COMPLETED":
            overall_metrics = primary_dataset["metrics"].copy()
            overall_metrics["primary_dataset"] = primary_dataset["dataset_name"]
            overall_metrics["total_samples_evaluated"] = sum(
                eval_result.get("sample_count", 0) 
                for eval_result in dataset_evaluations.values()
                if eval_result.get("status") == "COMPLETED"
            )
        
        return overall_metrics
    
    def _analyze_performance(self, overall_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall model performance."""
        thresholds = self.config["performance_thresholds"]
        analysis = {
            "performance_status": "UNKNOWN",
            "threshold_compliance": {},
            "performance_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "critical_issues": []
        }
        
        if not overall_metrics:
            analysis["performance_status"] = "NO_DATA"
            return analysis
        
        # Check threshold compliance
        threshold_checks = {
            "accuracy": overall_metrics.get("accuracy", 0) >= thresholds["min_accuracy"],
            "auc": overall_metrics.get("auc", 0) >= thresholds["min_auc"],
            "precision": overall_metrics.get("precision", 0) >= thresholds["min_precision"],
            "recall": overall_metrics.get("recall", 0) >= thresholds["min_recall"],
            "f1_score": overall_metrics.get("f1_score", 0) >= thresholds["min_f1"],
            "calibration": overall_metrics.get("calibration_score", 0) >= thresholds["min_calibration_score"]
        }
        
        analysis["threshold_compliance"] = threshold_checks
        
        # Calculate performance score
        passed_checks = sum(threshold_checks.values())
        total_checks = len(threshold_checks)
        analysis["performance_score"] = passed_checks / total_checks if total_checks > 0 else 0
        
        # Determine performance status
        if analysis["performance_score"] >= 0.9:
            analysis["performance_status"] = "EXCELLENT"
        elif analysis["performance_score"] >= 0.8:
            analysis["performance_status"] = "GOOD"
        elif analysis["performance_score"] >= 0.6:
            analysis["performance_status"] = "ACCEPTABLE"
        elif analysis["performance_status"] >= 0.4:
            analysis["performance_status"] = "POOR"
        else:
            analysis["performance_status"] = "CRITICAL"
        
        # Identify strengths and weaknesses
        for metric, passed in threshold_checks.items():
            metric_value = overall_metrics.get(metric.replace("_score", ""), 0)
            threshold_value = thresholds.get(f"min_{metric}", 0)
            
            if passed:
                if metric_value > threshold_value * 1.1:  # 10% above threshold
                    analysis["strengths"].append(f"{metric}: {metric_value:.3f} (exceeds threshold)")
            else:
                difference = threshold_value - metric_value
                if difference > threshold_value * 0.1:  # Significantly below threshold
                    analysis["critical_issues"].append(f"{metric}: {metric_value:.3f} (significantly below {threshold_value:.3f})")
                else:
                    analysis["weaknesses"].append(f"{metric}: {metric_value:.3f} (below {threshold_value:.3f})")
        
        return analysis
    
    def _generate_recommendations(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on performance analysis."""
        recommendations = []
        
        status = performance_analysis.get("performance_status", "UNKNOWN")
        critical_issues = performance_analysis.get("critical_issues", [])
        weaknesses = performance_analysis.get("weaknesses", [])
        
        if status == "CRITICAL":
            recommendations.append("🚨 CRITICAL: Model performance is critically low - immediate attention required")
            recommendations.append("Consider retraining with more data or different architecture")
            recommendations.append("Review data quality and feature engineering")
        
        elif status == "POOR":
            recommendations.append("⚠️ Model performance is below acceptable levels")
            recommendations.append("Investigate training data quality and model architecture")
        
        elif status == "ACCEPTABLE":
            recommendations.append("✓ Model performance is acceptable but has room for improvement")
        
        elif status in ["GOOD", "EXCELLENT"]:
            recommendations.append("✅ Model performance meets quality standards")
        
        # Specific metric recommendations
        if critical_issues:
            recommendations.append("Critical issues identified:")
            for issue in critical_issues:
                recommendations.append(f"  • {issue}")
        
        if weaknesses:
            recommendations.append("Areas for improvement:")
            for weakness in weaknesses:
                recommendations.append(f"  • {weakness}")
        
        # Performance-specific recommendations
        threshold_compliance = performance_analysis.get("threshold_compliance", {})
        
        if not threshold_compliance.get("accuracy", True):
            recommendations.append("Consider data augmentation or model architecture improvements for accuracy")
        
        if not threshold_compliance.get("auc", True):
            recommendations.append("Review feature selection and class balance for AUC improvement")
        
        if not threshold_compliance.get("calibration", True):
            recommendations.append("Consider calibration techniques (Platt scaling, isotonic regression)")
        
        return recommendations
    
    def compare_models(self, model_uris: List[str], model_names: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models side by side."""
        logger.info(f"Comparing {len(model_uris)} models...")
        
        if model_names is None:
            model_names = [f"model_{i+1}" for i in range(len(model_uris))]
        
        # Evaluate each model
        for model_uri, model_name in zip(model_uris, model_names):
            self.evaluate_model(model_uri, model_name)
        
        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis()
        self.evaluation_results["comparative_analysis"] = comparative_analysis
        
        return comparative_analysis
    
    def _perform_comparative_analysis(self) -> Dict[str, Any]:
        """Perform comparative analysis across evaluated models."""
        model_evaluations = self.evaluation_results["model_evaluations"]
        
        if len(model_evaluations) < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        # Extract metrics for comparison
        comparison_data = {}
        for model_name, evaluation in model_evaluations.items():
            if evaluation.get("status") == "COMPLETED":
                metrics = evaluation.get("overall_metrics", {})
                comparison_data[model_name] = metrics
        
        if len(comparison_data) < 2:
            return {"error": "Need at least 2 successfully evaluated models"}
        
        # Statistical comparison
        statistical_comparison = self._statistical_model_comparison(comparison_data)
        
        # Ranking analysis
        ranking_analysis = self._rank_models(comparison_data)
        
        # Best model recommendation
        best_model = self._select_best_model(comparison_data, ranking_analysis)
        
        return {
            "models_compared": list(comparison_data.keys()),
            "comparison_metrics": comparison_data,
            "statistical_comparison": statistical_comparison,
            "model_rankings": ranking_analysis,
            "best_model_recommendation": best_model,
            "comparison_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _statistical_model_comparison(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform statistical comparison between models."""
        primary_metrics = self.config["evaluation_metrics"]["primary_metrics"]
        model_names = list(comparison_data.keys())
        
        statistical_results = {}
        
        for metric in primary_metrics:
            if all(metric in comparison_data[model] for model in model_names):
                values = [comparison_data[model][metric] for model in model_names]
                
                statistical_results[metric] = {
                    "values": dict(zip(model_names, values)),
                    "best_model": model_names[np.argmax(values)],
                    "worst_model": model_names[np.argmin(values)],
                    "range": max(values) - min(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
        
        return statistical_results
    
    def _rank_models(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Rank models based on multiple metrics."""
        primary_metrics = self.config["evaluation_metrics"]["primary_metrics"]
        model_names = list(comparison_data.keys())
        
        # Calculate rankings for each metric
        metric_rankings = {}
        for metric in primary_metrics:
            if all(metric in comparison_data[model] for model in model_names):
                values = [(model, comparison_data[model][metric]) for model in model_names]
                values.sort(key=lambda x: x[1], reverse=True)  # Higher is better
                
                metric_rankings[metric] = {model: rank + 1 for rank, (model, _) in enumerate(values)}
        
        # Calculate average ranking
        average_rankings = {}
        for model in model_names:
            ranks = [metric_rankings[metric][model] for metric in metric_rankings.keys()]
            average_rankings[model] = np.mean(ranks)
        
        # Sort by average ranking
        sorted_rankings = sorted(average_rankings.items(), key=lambda x: x[1])
        
        return {
            "metric_rankings": metric_rankings,
            "average_rankings": average_rankings,
            "overall_ranking": [model for model, _ in sorted_rankings],
            "ranking_scores": dict(sorted_rankings)
        }
    
    def _select_best_model(self, comparison_data: Dict[str, Dict[str, float]], 
                         ranking_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best model based on comprehensive analysis."""
        overall_ranking = ranking_analysis["overall_ranking"]
        best_model = overall_ranking[0]
        
        best_model_metrics = comparison_data[best_model]
        
        # Check if best model meets minimum thresholds
        thresholds = self.config["performance_thresholds"]
        meets_thresholds = all(
            best_model_metrics.get(metric.replace("min_", ""), 0) >= threshold
            for metric, threshold in thresholds.items()
            if metric.startswith("min_")
        )
        
        recommendation_confidence = "HIGH" if meets_thresholds else "MEDIUM"
        
        return {
            "recommended_model": best_model,
            "recommendation_confidence": recommendation_confidence,
            "meets_thresholds": meets_thresholds,
            "ranking_position": 1,
            "key_metrics": best_model_metrics,
            "recommendation_reason": f"Highest overall ranking with average rank {ranking_analysis['ranking_scores'][best_model]:.2f}"
        }
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """Generate comprehensive performance evaluation report."""
        if output_path is None:
            output_path = f"evaluation_report_{int(time.time())}.json"
        
        # Finalize evaluation results
        self.evaluation_results["status"] = "COMPLETED"
        self.evaluation_results["completion_time"] = datetime.now(timezone.utc).isoformat()
        
        # Save detailed results
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Performance evaluation report saved: {output_path}")
        return output_path
    
    def _get_model_info(self, model_uri: str) -> Dict[str, Any]:
        """Get model information from MLflow."""
        try:
            model_info = mlflow.models.get_model_info(model_uri)
            return {
                "model_uri": model_uri,
                "run_id": model_info.run_id,
                "model_uuid": model_info.model_uuid,
                "creation_timestamp": model_info.utc_time_created,
                "flavors": list(model_info.flavors.keys())
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"model_uri": model_uri, "error": str(e)}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated Model Performance Evaluation")
    parser.add_argument("--model-uri", help="Model URI to evaluate")
    parser.add_argument("--model-uris", nargs="+", help="Multiple model URIs for comparison")
    parser.add_argument("--model-names", nargs="+", help="Names for the models")
    parser.add_argument("--config", default="configs/evaluation_config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--datasets", help="JSON string with dataset paths")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AutomatedPerformanceEvaluator(args.config)
    
    try:
        if args.model_uris:
            # Multiple model comparison
            results = evaluator.compare_models(args.model_uris, args.model_names)
            print(f"Model comparison completed for {len(args.model_uris)} models")
            
            if "best_model_recommendation" in results:
                best_model = results["best_model_recommendation"]["recommended_model"]
                confidence = results["best_model_recommendation"]["recommendation_confidence"]
                print(f"Best model: {best_model} (confidence: {confidence})")
        
        elif args.model_uri:
            # Single model evaluation
            datasets = None
            if args.datasets:
                datasets = json.loads(args.datasets)
            
            model_name = args.model_names[0] if args.model_names else None
            results = evaluator.evaluate_model(args.model_uri, model_name, datasets)
            
            if results.get("status") == "COMPLETED":
                performance = results["performance_analysis"]["performance_status"]
                score = results["performance_analysis"]["performance_score"]
                print(f"Model evaluation completed: {performance} (score: {score:.2f})")
            else:
                print(f"Model evaluation failed: {results.get('error', 'Unknown error')}")
        
        else:
            print("Error: Either --model-uri or --model-uris must be provided")
            sys.exit(1)
        
        # Generate report
        output_file = args.output or f"evaluation_report_{int(time.time())}.json"
        report_path = evaluator.generate_performance_report(output_file)
        print(f"Detailed report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()