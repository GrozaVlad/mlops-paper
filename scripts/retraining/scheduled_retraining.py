#!/usr/bin/env python3
"""
Scheduled Model Retraining Pipeline
Automated pipeline for periodic model retraining with data drift detection,
performance evaluation, and model promotion workflows.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated model retraining pipeline with drift detection and evaluation."""
    
    def __init__(self, config_path: str = "configs/retraining_config.yaml"):
        """Initialize retraining pipeline.
        
        Args:
            config_path: Path to retraining configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Initialize tracking
        self.retraining_run_id = None
        self.baseline_model_uri = None
        self.current_model_uri = None
        
        # Results storage
        self.retraining_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "trigger_reason": None,
            "data_analysis": {},
            "training_results": {},
            "evaluation_results": {},
            "promotion_decision": {},
            "end_time": None,
            "status": "RUNNING"
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load retraining configuration."""
        # Default configuration if file doesn't exist
        default_config = {
            "retraining_schedule": {
                "frequency": "weekly",  # daily, weekly, monthly
                "day_of_week": 0,  # 0=Monday
                "hour": 2,  # 2 AM UTC
                "max_training_time_hours": 6
            },
            "trigger_conditions": {
                "performance_degradation_threshold": 0.05,  # 5% drop
                "data_drift_threshold": 0.3,
                "concept_drift_threshold": 0.2,
                "min_new_data_samples": 1000,
                "force_retrain_days": 30
            },
            "training_config": {
                "experiment_name": "automated_retraining",
                "model_name": "DrugBAN",
                "training_epochs": 50,
                "early_stopping_patience": 10,
                "validation_split": 0.2,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "evaluation_criteria": {
                "min_accuracy": 0.75,
                "min_auc": 0.80,
                "max_performance_drop": 0.02,  # 2% worse than baseline
                "statistical_significance_alpha": 0.05
            },
            "promotion_rules": {
                "auto_promote_threshold": 0.05,  # 5% improvement
                "require_manual_approval": True,
                "staging_duration_hours": 24,
                "rollback_on_production_issues": True
            }
        }
        
        try:
            import yaml
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Merge with defaults
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
    
    def should_trigger_retraining(self) -> Tuple[bool, List[str]]:
        """Determine if retraining should be triggered.
        
        Returns:
            Tuple of (should_retrain, reasons)
        """
        logger.info("Evaluating retraining trigger conditions...")
        
        reasons = []
        should_retrain = False
        
        # Check 1: Scheduled retraining
        if self._is_scheduled_time():
            reasons.append("Scheduled retraining time reached")
            should_retrain = True
        
        # Check 2: Performance degradation
        performance_drop = self._check_performance_degradation()
        if performance_drop > self.config["trigger_conditions"]["performance_degradation_threshold"]:
            reasons.append(f"Performance degradation detected: {performance_drop:.3f}")
            should_retrain = True
        
        # Check 3: Data drift
        data_drift_score = self._check_data_drift()
        if data_drift_score > self.config["trigger_conditions"]["data_drift_threshold"]:
            reasons.append(f"Data drift detected: {data_drift_score:.3f}")
            should_retrain = True
        
        # Check 4: Concept drift
        concept_drift_score = self._check_concept_drift()
        if concept_drift_score > self.config["trigger_conditions"]["concept_drift_threshold"]:
            reasons.append(f"Concept drift detected: {concept_drift_score:.3f}")
            should_retrain = True
        
        # Check 5: Sufficient new data
        new_data_count = self._check_new_data_availability()
        if new_data_count < self.config["trigger_conditions"]["min_new_data_samples"]:
            if should_retrain:
                reasons.append(f"Warning: Limited new data available ({new_data_count} samples)")
        else:
            reasons.append(f"Sufficient new data available: {new_data_count} samples")
        
        # Check 6: Force retrain after max days
        days_since_last_train = self._days_since_last_training()
        if days_since_last_train >= self.config["trigger_conditions"]["force_retrain_days"]:
            reasons.append(f"Force retrain: {days_since_last_train} days since last training")
            should_retrain = True
        
        logger.info(f"Retraining decision: {should_retrain}")
        for reason in reasons:
            logger.info(f"  - {reason}")
        
        return should_retrain, reasons
    
    def _is_scheduled_time(self) -> bool:
        """Check if current time matches retraining schedule."""
        schedule = self.config["retraining_schedule"]
        now = datetime.now(timezone.utc)
        
        if schedule["frequency"] == "daily":
            return now.hour == schedule["hour"]
        elif schedule["frequency"] == "weekly":
            return (now.weekday() == schedule["day_of_week"] and 
                   now.hour == schedule["hour"])
        elif schedule["frequency"] == "monthly":
            return (now.day == 1 and now.hour == schedule["hour"])
        
        return False
    
    def _check_performance_degradation(self) -> float:
        """Check for model performance degradation."""
        try:
            # Get current production model performance
            current_model = self._get_current_production_model()
            if not current_model:
                return 0.0
            
            # Get recent performance metrics
            recent_metrics = self._get_recent_performance_metrics()
            baseline_metrics = self._get_baseline_performance_metrics(current_model)
            
            if not recent_metrics or not baseline_metrics:
                return 0.0
            
            # Calculate performance drop
            current_accuracy = recent_metrics.get("accuracy", 0)
            baseline_accuracy = baseline_metrics.get("accuracy", 0)
            
            performance_drop = max(0, baseline_accuracy - current_accuracy)
            
            self.retraining_results["data_analysis"]["performance_degradation"] = {
                "current_accuracy": current_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "performance_drop": performance_drop
            }
            
            return performance_drop
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return 0.0
    
    def _check_data_drift(self) -> float:
        """Check for data drift using statistical methods."""
        try:
            # Load reference and current data
            reference_data = self._load_reference_data()
            current_data = self._load_current_data()
            
            if reference_data is None or current_data is None:
                return 0.0
            
            # Simple drift detection using feature distribution comparison
            drift_scores = []
            
            # Compare distributions for numerical features
            for column in reference_data.select_dtypes(include=[np.number]).columns:
                if column in current_data.columns:
                    # Kolmogorov-Smirnov test
                    from scipy import stats
                    ks_stat, p_value = stats.ks_2samp(
                        reference_data[column].dropna(),
                        current_data[column].dropna()
                    )
                    drift_scores.append(ks_stat)
            
            avg_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            
            self.retraining_results["data_analysis"]["data_drift"] = {
                "drift_score": avg_drift_score,
                "features_analyzed": len(drift_scores),
                "reference_samples": len(reference_data),
                "current_samples": len(current_data)
            }
            
            return avg_drift_score
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return 0.0
    
    def _check_concept_drift(self) -> float:
        """Check for concept drift by comparing prediction distributions."""
        try:
            # This would compare prediction patterns over time
            # For demonstration, using a simulated drift score
            concept_drift_score = np.random.uniform(0, 0.4)
            
            self.retraining_results["data_analysis"]["concept_drift"] = {
                "drift_score": concept_drift_score,
                "method": "prediction_distribution_comparison"
            }
            
            return concept_drift_score
            
        except Exception as e:
            logger.error(f"Error checking concept drift: {e}")
            return 0.0
    
    def _check_new_data_availability(self) -> int:
        """Check how much new data is available for training."""
        try:
            # Check for new data since last training
            last_training_date = self._get_last_training_date()
            new_data_count = self._count_data_since_date(last_training_date)
            
            self.retraining_results["data_analysis"]["new_data"] = {
                "samples_available": new_data_count,
                "last_training_date": last_training_date.isoformat() if last_training_date else None
            }
            
            return new_data_count
            
        except Exception as e:
            logger.error(f"Error checking new data availability: {e}")
            return 0
    
    def _days_since_last_training(self) -> int:
        """Calculate days since last training."""
        try:
            last_training_date = self._get_last_training_date()
            if last_training_date:
                return (datetime.now(timezone.utc) - last_training_date).days
            return 999  # Force retrain if no previous training found
        except Exception as e:
            logger.error(f"Error calculating days since last training: {e}")
            return 999
    
    def execute_retraining(self, trigger_reasons: List[str]) -> Dict[str, Any]:
        """Execute the retraining pipeline.
        
        Args:
            trigger_reasons: List of reasons that triggered retraining
            
        Returns:
            Retraining results
        """
        logger.info("Starting automated model retraining...")
        
        self.retraining_results["trigger_reason"] = trigger_reasons
        
        try:
            # Step 1: Prepare data
            logger.info("Step 1: Preparing training data...")
            train_data, val_data, test_data = self._prepare_training_data()
            
            # Step 2: Setup MLflow experiment
            experiment_name = self.config["training_config"]["experiment_name"]
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            # Step 3: Train new model
            logger.info("Step 2: Training new model...")
            with mlflow.start_run(run_name=f"automated_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                self.retraining_run_id = run.info.run_id
                
                # Log trigger reasons
                mlflow.log_param("trigger_reasons", ", ".join(trigger_reasons))
                mlflow.log_param("training_type", "automated_retraining")
                
                # Train model
                new_model, training_metrics = self._train_model(train_data, val_data)
                
                # Log training metrics
                for metric_name, value in training_metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Step 4: Evaluate new model
                logger.info("Step 3: Evaluating new model...")
                evaluation_results = self._evaluate_model(new_model, test_data)
                
                # Log evaluation metrics
                for metric_name, value in evaluation_results.items():
                    mlflow.log_metric(f"eval_{metric_name}", value)
                
                # Step 5: Compare with baseline
                logger.info("Step 4: Comparing with baseline model...")
                comparison_results = self._compare_with_baseline(evaluation_results)
                
                # Step 6: Make promotion decision
                logger.info("Step 5: Making promotion decision...")
                promotion_decision = self._make_promotion_decision(comparison_results)
                
                # Log model if it meets criteria
                if promotion_decision["should_promote"]:
                    model_uri = mlflow.pytorch.log_model(
                        new_model,
                        "model",
                        registered_model_name=self.config["training_config"]["model_name"]
                    ).model_uri
                    self.current_model_uri = model_uri
                
                # Update results
                self.retraining_results["training_results"] = training_metrics
                self.retraining_results["evaluation_results"] = evaluation_results
                self.retraining_results["promotion_decision"] = promotion_decision
                self.retraining_results["status"] = "COMPLETED"
                
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            self.retraining_results["status"] = "FAILED"
            self.retraining_results["error"] = str(e)
            raise
        
        finally:
            self.retraining_results["end_time"] = datetime.now(timezone.utc).isoformat()
        
        return self.retraining_results
    
    def _prepare_training_data(self) -> Tuple[Any, Any, Any]:
        """Prepare training, validation, and test data."""
        # Load and prepare data (simplified for demonstration)
        logger.info("Loading and preparing training data...")
        
        # This would load actual drug-target interaction data
        # For demonstration, creating sample data
        n_samples = 5000
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        
        logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return train_data, val_data, test_data
    
    def _train_model(self, train_data: Tuple, val_data: Tuple) -> Tuple[Any, Dict[str, float]]:
        """Train a new model."""
        logger.info("Training new model...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Simple neural network for demonstration
        class SimpleModel(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x
        
        # Initialize model
        model = SimpleModel(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["training_config"]["learning_rate"])
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config["training_config"]["training_epochs"]):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
                # Calculate accuracy
                val_predictions = (val_outputs > 0.5).float()
                val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["training_config"]["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        training_metrics = {
            "final_train_loss": loss.item(),
            "final_val_loss": best_val_loss.item(),
            "final_val_accuracy": val_accuracy,
            "epochs_trained": epoch + 1
        }
        
        logger.info("Model training completed")
        return model, training_metrics
    
    def _evaluate_model(self, model: Any, test_data: Tuple) -> Dict[str, float]:
        """Evaluate model on test data."""
        logger.info("Evaluating model on test data...")
        
        X_test, y_test = test_data
        X_test_tensor = torch.FloatTensor(X_test)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predictions = (test_outputs > 0.5).float().numpy()
            probabilities = test_outputs.numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='binary')
        recall = recall_score(y_test, predictions, average='binary')
        f1 = f1_score(y_test, predictions, average='binary')
        auc = roc_auc_score(y_test, probabilities)
        
        evaluation_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_score": auc,
            "test_samples": len(y_test)
        }
        
        logger.info(f"Evaluation completed: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
        return evaluation_results
    
    def _compare_with_baseline(self, evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """Compare new model with baseline model."""
        logger.info("Comparing with baseline model...")
        
        # Get baseline model performance
        baseline_model = self._get_current_production_model()
        baseline_metrics = self._get_baseline_performance_metrics(baseline_model)
        
        if not baseline_metrics:
            logger.warning("No baseline model found, treating as first model")
            return {
                "baseline_available": False,
                "performance_improvement": evaluation_results,
                "recommendation": "PROMOTE"
            }
        
        # Calculate improvements
        improvements = {}
        for metric_name, new_value in evaluation_results.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                improvement = new_value - baseline_value
                improvements[f"{metric_name}_improvement"] = improvement
                improvements[f"{metric_name}_improvement_pct"] = (improvement / baseline_value) * 100 if baseline_value > 0 else 0
        
        # Determine recommendation
        accuracy_improvement = improvements.get("accuracy_improvement", 0)
        auc_improvement = improvements.get("auc_improvement", 0)
        
        if accuracy_improvement >= self.config["promotion_rules"]["auto_promote_threshold"]:
            recommendation = "PROMOTE"
        elif accuracy_improvement >= -self.config["evaluation_criteria"]["max_performance_drop"]:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"
        
        comparison_results = {
            "baseline_available": True,
            "baseline_metrics": baseline_metrics,
            "new_metrics": evaluation_results,
            "improvements": improvements,
            "recommendation": recommendation
        }
        
        logger.info(f"Comparison completed: Recommendation={recommendation}")
        return comparison_results
    
    def _make_promotion_decision(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Make model promotion decision based on evaluation results."""
        logger.info("Making model promotion decision...")
        
        recommendation = comparison_results.get("recommendation", "REJECT")
        should_promote = False
        promotion_reason = ""
        
        # Check evaluation criteria
        eval_results = comparison_results.get("new_metrics", {})
        
        meets_accuracy = eval_results.get("accuracy", 0) >= self.config["evaluation_criteria"]["min_accuracy"]
        meets_auc = eval_results.get("auc_score", 0) >= self.config["evaluation_criteria"]["min_auc"]
        
        if not meets_accuracy:
            promotion_reason = f"Accuracy {eval_results.get('accuracy', 0):.3f} below minimum {self.config['evaluation_criteria']['min_accuracy']}"
        elif not meets_auc:
            promotion_reason = f"AUC {eval_results.get('auc_score', 0):.3f} below minimum {self.config['evaluation_criteria']['min_auc']}"
        elif recommendation == "PROMOTE":
            should_promote = True
            promotion_reason = "Model meets all criteria and shows improvement"
        elif recommendation == "REVIEW":
            should_promote = self.config["promotion_rules"]["require_manual_approval"] == False
            promotion_reason = "Model performance stable, manual review recommended"
        else:
            promotion_reason = "Model performance degraded significantly"
        
        promotion_decision = {
            "should_promote": should_promote,
            "promotion_reason": promotion_reason,
            "meets_accuracy_criteria": meets_accuracy,
            "meets_auc_criteria": meets_auc,
            "recommendation": recommendation,
            "requires_manual_approval": self.config["promotion_rules"]["require_manual_approval"] and should_promote,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Promotion decision: {should_promote} - {promotion_reason}")
        return promotion_decision
    
    # Helper methods (simplified implementations)
    def _get_current_production_model(self):
        """Get current production model."""
        try:
            model_name = self.config["training_config"]["model_name"]
            latest_versions = self.mlflow_client.get_latest_versions(model_name, stages=["Production"])
            return latest_versions[0] if latest_versions else None
        except Exception:
            return None
    
    def _get_recent_performance_metrics(self) -> Optional[Dict[str, float]]:
        """Get recent performance metrics from monitoring."""
        # This would query monitoring system for recent metrics
        # For demonstration, return sample metrics
        return {
            "accuracy": 0.82,
            "precision": 0.80,
            "recall": 0.84,
            "f1_score": 0.82,
            "auc_score": 0.88
        }
    
    def _get_baseline_performance_metrics(self, model) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics."""
        if not model:
            return None
        
        # This would retrieve metrics from model registry
        # For demonstration, return sample baseline metrics
        return {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "auc_score": 0.91
        }
    
    def _load_reference_data(self):
        """Load reference data for drift detection."""
        # This would load reference dataset
        # For demonstration, return sample data
        return pd.DataFrame(np.random.randn(1000, 50))
    
    def _load_current_data(self):
        """Load current data for drift detection."""
        # This would load recent data
        # For demonstration, return sample data with slight drift
        return pd.DataFrame(np.random.randn(500, 50) + 0.1)
    
    def _get_last_training_date(self) -> Optional[datetime]:
        """Get date of last training."""
        try:
            model_name = self.config["training_config"]["model_name"]
            latest_versions = self.mlflow_client.get_latest_versions(model_name)
            if latest_versions:
                run = self.mlflow_client.get_run(latest_versions[0].run_id)
                return datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
            return None
        except Exception:
            return None
    
    def _count_data_since_date(self, since_date: Optional[datetime]) -> int:
        """Count new data samples since date."""
        # This would query data store for new samples
        # For demonstration, return random count
        return np.random.randint(500, 2000)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated Model Retraining Pipeline")
    parser.add_argument("--config", default="configs/retraining_config.yaml", 
                       help="Retraining configuration file")
    parser.add_argument("--force", action="store_true", 
                       help="Force retraining regardless of conditions")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Check conditions without executing retraining")
    parser.add_argument("--output", help="Output file for retraining results")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RetrainingPipeline(args.config)
    
    # Check trigger conditions
    should_retrain, trigger_reasons = pipeline.should_trigger_retraining()
    
    if args.force:
        should_retrain = True
        trigger_reasons.append("Force retraining requested")
    
    if args.dry_run:
        print(f"Dry run - Retraining would be triggered: {should_retrain}")
        for reason in trigger_reasons:
            print(f"  - {reason}")
        sys.exit(0)
    
    if not should_retrain:
        print("No retraining triggers detected")
        sys.exit(0)
    
    # Execute retraining
    try:
        results = pipeline.execute_retraining(trigger_reasons)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Print summary
        print(f"Retraining completed: {results['status']}")
        print(f"Promotion decision: {results['promotion_decision']['should_promote']}")
        print(f"Reason: {results['promotion_decision']['promotion_reason']}")
        
        if results["status"] == "COMPLETED":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()