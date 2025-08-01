#!/usr/bin/env python3
"""
A/B Testing Framework for Model Updates
Comprehensive framework for comparing model performance through controlled experiments.
"""

import argparse
import json
import sys
import time
import os
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import redis
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABTestFramework:
    """A/B testing framework for model performance comparison."""
    
    def __init__(self, config_path: str = "configs/ab_test_config.yaml"):
        """Initialize A/B testing framework.
        
        Args:
            config_path: Path to A/B test configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        # Redis client for experiment tracking
        self.redis_client = self._setup_redis()
        
        # Experiment tracking
        self.experiment_id = None
        self.experiment_state = "INACTIVE"
        self.traffic_allocator = TrafficAllocator(self.config)
        self.metrics_collector = ABTestMetricsCollector(self.config, self.redis_client)
        
        # Results storage
        self.ab_test_results = {
            "experiment_id": None,
            "start_time": None,
            "end_time": None,
            "status": "INACTIVE",
            "models": {},
            "traffic_allocation": {},
            "metrics": {},
            "statistical_analysis": {},
            "recommendation": {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load A/B test configuration."""
        default_config = {
            "experiment_config": {
                "default_duration_hours": 72,
                "min_sample_size": 1000,
                "statistical_power": 0.8,
                "significance_level": 0.05,
                "effect_size": 0.02  # 2% improvement
            },
            "traffic_allocation": {
                "default_split": {"champion": 0.9, "challenger": 0.1},
                "ramp_up_strategy": "gradual",  # gradual, immediate
                "max_challenger_traffic": 0.5,
                "ramp_up_duration_hours": 24
            },
            "metrics_tracking": {
                "primary_metrics": ["accuracy", "auc", "precision", "recall"],
                "secondary_metrics": ["latency", "throughput", "error_rate"],
                "business_metrics": ["user_satisfaction", "prediction_confidence"],
                "collection_interval_seconds": 60
            },
            "guardrail_metrics": {
                "max_error_rate": 0.05,
                "max_latency_increase": 0.2,  # 20% increase
                "min_accuracy": 0.75
            },
            "early_stopping": {
                "enable": True,
                "check_interval_hours": 6,
                "min_runtime_hours": 12,
                "stop_conditions": {
                    "significance_achieved": True,
                    "guardrail_violation": True,
                    "sample_size_reached": True
                }
            },
            "redis_config": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "key_prefix": "ab_test:"
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
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis client for experiment tracking."""
        try:
            redis_config = self.config.get("redis_config", {})
            client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                decode_responses=True
            )
            # Test connection
            client.ping()
            logger.info("Redis connection established")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory tracking")
            return None
    
    def create_experiment(self, champion_model_uri: str, challenger_model_uri: str,
                         experiment_name: str, duration_hours: int = None,
                         traffic_split: Dict[str, float] = None) -> str:
        """Create a new A/B test experiment.
        
        Args:
            champion_model_uri: URI of the current production model
            challenger_model_uri: URI of the new model to test
            experiment_name: Name for the experiment
            duration_hours: Duration of the experiment
            traffic_split: Traffic allocation between models
            
        Returns:
            Experiment ID
        """
        logger.info(f"Creating A/B test experiment: {experiment_name}")
        
        # Generate experiment ID
        experiment_id = f"ab_test_{int(time.time())}_{random.randint(1000, 9999)}"
        self.experiment_id = experiment_id
        
        # Default values
        if duration_hours is None:
            duration_hours = self.config["experiment_config"]["default_duration_hours"]
        
        if traffic_split is None:
            traffic_split = self.config["traffic_allocation"]["default_split"]
        
        # Validate models
        champion_info = self._get_model_info(champion_model_uri)
        challenger_info = self._get_model_info(challenger_model_uri)
        
        if not champion_info or not challenger_info:
            raise ValueError("Unable to retrieve model information")
        
        # Setup experiment configuration
        experiment_config = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "duration_hours": duration_hours,
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=duration_hours)).isoformat(),
            "status": "CREATED",
            "models": {
                "champion": {
                    "model_uri": champion_model_uri,
                    "model_info": champion_info,
                    "allocation": traffic_split.get("champion", 0.9)
                },
                "challenger": {
                    "model_uri": challenger_model_uri,
                    "model_info": challenger_info,
                    "allocation": traffic_split.get("challenger", 0.1)
                }
            },
            "traffic_allocation": traffic_split,
            "created_by": os.getenv("USER", "system"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store experiment configuration
        self._store_experiment_config(experiment_id, experiment_config)
        
        # Initialize metrics tracking
        self.metrics_collector.initialize_experiment(experiment_id, experiment_config)
        
        # Update results
        self.ab_test_results.update({
            "experiment_id": experiment_id,
            "start_time": experiment_config["start_time"],
            "status": "CREATED",
            "models": experiment_config["models"],
            "traffic_allocation": traffic_split
        })
        
        logger.info(f"A/B test experiment created: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str = None) -> bool:
        """Start the A/B test experiment.
        
        Args:
            experiment_id: Experiment ID (uses current if not specified)
            
        Returns:
            Success status
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        if not experiment_id:
            raise ValueError("No experiment ID specified")
        
        logger.info(f"Starting A/B test experiment: {experiment_id}")
        
        try:
            # Load experiment configuration
            experiment_config = self._load_experiment_config(experiment_id)
            if not experiment_config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Validate readiness
            if not self._validate_experiment_readiness(experiment_config):
                raise ValueError("Experiment not ready to start")
            
            # Start traffic allocation
            self.traffic_allocator.start_allocation(experiment_id, experiment_config)
            
            # Start metrics collection
            self.metrics_collector.start_collection(experiment_id)
            
            # Update experiment status
            experiment_config["status"] = "RUNNING"
            experiment_config["actual_start_time"] = datetime.now(timezone.utc).isoformat()
            self._store_experiment_config(experiment_id, experiment_config)
            
            # Update tracking
            self.experiment_state = "RUNNING"
            self.ab_test_results["status"] = "RUNNING"
            
            logger.info(f"A/B test experiment started successfully: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str = None, reason: str = "Manual stop") -> bool:
        """Stop the A/B test experiment.
        
        Args:
            experiment_id: Experiment ID (uses current if not specified)
            reason: Reason for stopping
            
        Returns:
            Success status
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        if not experiment_id:
            raise ValueError("No experiment ID specified")
        
        logger.info(f"Stopping A/B test experiment: {experiment_id}, Reason: {reason}")
        
        try:
            # Load experiment configuration
            experiment_config = self._load_experiment_config(experiment_id)
            if not experiment_config:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Stop traffic allocation
            self.traffic_allocator.stop_allocation(experiment_id)
            
            # Stop metrics collection
            self.metrics_collector.stop_collection(experiment_id)
            
            # Update experiment status
            experiment_config["status"] = "STOPPED"
            experiment_config["stop_time"] = datetime.now(timezone.utc).isoformat()
            experiment_config["stop_reason"] = reason
            self._store_experiment_config(experiment_id, experiment_config)
            
            # Update tracking
            self.experiment_state = "STOPPED"
            self.ab_test_results["status"] = "STOPPED"
            self.ab_test_results["end_time"] = experiment_config["stop_time"]
            
            logger.info(f"A/B test experiment stopped: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {e}")
            return False
    
    def analyze_results(self, experiment_id: str = None) -> Dict[str, Any]:
        """Analyze A/B test results and provide recommendations.
        
        Args:
            experiment_id: Experiment ID (uses current if not specified)
            
        Returns:
            Analysis results
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        if not experiment_id:
            raise ValueError("No experiment ID specified")
        
        logger.info(f"Analyzing A/B test results: {experiment_id}")
        
        # Load experiment configuration and metrics
        experiment_config = self._load_experiment_config(experiment_id)
        metrics_data = self.metrics_collector.get_experiment_metrics(experiment_id)
        
        if not experiment_config or not metrics_data:
            raise ValueError("Insufficient data for analysis")
        
        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(metrics_data)
        
        # Check guardrail metrics
        guardrail_results = self._check_guardrail_violations(metrics_data)
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(metrics_data, statistical_results)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            statistical_results, guardrail_results, business_impact
        )
        
        # Compile analysis results
        analysis_results = {
            "experiment_id": experiment_id,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_duration": self._calculate_experiment_duration(experiment_config),
            "sample_sizes": self._calculate_sample_sizes(metrics_data),
            "statistical_analysis": statistical_results,
            "guardrail_analysis": guardrail_results,
            "business_impact": business_impact,
            "recommendation": recommendation,
            "confidence_level": self._calculate_confidence_level(statistical_results),
            "next_steps": self._generate_next_steps(recommendation)
        }
        
        # Store analysis results
        self._store_analysis_results(experiment_id, analysis_results)
        
        # Update tracking
        self.ab_test_results["statistical_analysis"] = statistical_results
        self.ab_test_results["recommendation"] = recommendation
        
        logger.info(f"A/B test analysis completed: {recommendation['decision']}")
        return analysis_results
    
    def _perform_statistical_analysis(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on A/B test metrics."""
        logger.info("Performing statistical analysis...")
        
        analysis_results = {}
        primary_metrics = self.config["metrics_tracking"]["primary_metrics"]
        
        for metric in primary_metrics:
            if metric in metrics_data:
                champion_values = metrics_data[metric].get("champion", [])
                challenger_values = metrics_data[metric].get("challenger", [])
                
                if len(champion_values) > 0 and len(challenger_values) > 0:
                    # T-test for metric comparison
                    t_stat, p_value = stats.ttest_ind(challenger_values, champion_values)
                    
                    # Effect size (Cohen's d)
                    champion_mean = np.mean(champion_values)
                    challenger_mean = np.mean(challenger_values)
                    pooled_std = np.sqrt(
                        ((len(champion_values) - 1) * np.var(champion_values, ddof=1) +
                         (len(challenger_values) - 1) * np.var(challenger_values, ddof=1)) /
                        (len(champion_values) + len(challenger_values) - 2)
                    )
                    cohens_d = (challenger_mean - champion_mean) / pooled_std if pooled_std > 0 else 0
                    
                    # Confidence interval
                    confidence_level = 1 - self.config["experiment_config"]["significance_level"]
                    se_diff = pooled_std * np.sqrt(1/len(champion_values) + 1/len(challenger_values))
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, 
                                           len(champion_values) + len(challenger_values) - 2)
                    margin_error = t_critical * se_diff
                    
                    improvement = challenger_mean - champion_mean
                    ci_lower = improvement - margin_error
                    ci_upper = improvement + margin_error
                    
                    analysis_results[metric] = {
                        "champion_mean": champion_mean,
                        "challenger_mean": challenger_mean,
                        "improvement": improvement,
                        "improvement_pct": (improvement / champion_mean * 100) if champion_mean > 0 else 0,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "is_significant": p_value < self.config["experiment_config"]["significance_level"],
                        "cohens_d": cohens_d,
                        "effect_size": self._interpret_effect_size(abs(cohens_d)),
                        "confidence_interval": {
                            "lower": ci_lower,
                            "upper": ci_upper,
                            "level": confidence_level
                        },
                        "sample_sizes": {
                            "champion": len(champion_values),
                            "challenger": len(challenger_values)
                        }
                    }
        
        return analysis_results
    
    def _check_guardrail_violations(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for guardrail metric violations."""
        logger.info("Checking guardrail violations...")
        
        guardrail_config = self.config["guardrail_metrics"]
        violations = []
        guardrail_status = "PASSED"
        
        # Check error rate
        if "error_rate" in metrics_data:
            challenger_error_rate = np.mean(metrics_data["error_rate"].get("challenger", [0]))
            if challenger_error_rate > guardrail_config["max_error_rate"]:
                violations.append({
                    "metric": "error_rate",
                    "value": challenger_error_rate,
                    "threshold": guardrail_config["max_error_rate"],
                    "severity": "HIGH"
                })
                guardrail_status = "VIOLATED"
        
        # Check latency increase
        if "latency" in metrics_data:
            champion_latency = np.mean(metrics_data["latency"].get("champion", [0]))
            challenger_latency = np.mean(metrics_data["latency"].get("challenger", [0]))
            
            if champion_latency > 0:
                latency_increase = (challenger_latency - champion_latency) / champion_latency
                if latency_increase > guardrail_config["max_latency_increase"]:
                    violations.append({
                        "metric": "latency",
                        "value": latency_increase,
                        "threshold": guardrail_config["max_latency_increase"],
                        "severity": "MEDIUM"
                    })
                    guardrail_status = "VIOLATED"
        
        # Check minimum accuracy
        if "accuracy" in metrics_data:
            challenger_accuracy = np.mean(metrics_data["accuracy"].get("challenger", [0]))
            if challenger_accuracy < guardrail_config["min_accuracy"]:
                violations.append({
                    "metric": "accuracy",
                    "value": challenger_accuracy,
                    "threshold": guardrail_config["min_accuracy"],
                    "severity": "HIGH"
                })
                guardrail_status = "VIOLATED"
        
        return {
            "status": guardrail_status,
            "violations": violations,
            "total_violations": len(violations),
            "high_severity_violations": len([v for v in violations if v["severity"] == "HIGH"])
        }
    
    def _calculate_business_impact(self, metrics_data: Dict[str, Any], 
                                 statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact of the model change."""
        logger.info("Calculating business impact...")
        
        # This would calculate actual business metrics
        # For demonstration, using simplified calculations
        
        accuracy_improvement = statistical_results.get("accuracy", {}).get("improvement", 0)
        latency_change = statistical_results.get("latency", {}).get("improvement", 0)
        
        # Estimate impact on business metrics
        estimated_user_impact = accuracy_improvement * 1000  # Estimated additional correct predictions
        estimated_cost_impact = -latency_change * 0.001  # Simplified cost calculation
        
        return {
            "estimated_accuracy_gain": accuracy_improvement,
            "estimated_user_impact": estimated_user_impact,
            "estimated_cost_impact": estimated_cost_impact,
            "confidence": "medium",  # Based on statistical significance
            "projection_period": "monthly"
        }
    
    def _generate_recommendation(self, statistical_results: Dict[str, Any],
                               guardrail_results: Dict[str, Any],
                               business_impact: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation based on analysis results."""
        logger.info("Generating recommendation...")
        
        # Check for guardrail violations first
        if guardrail_results["status"] == "VIOLATED":
            if guardrail_results["high_severity_violations"] > 0:
                decision = "REJECT"
                reason = "High severity guardrail violations detected"
                confidence = "HIGH"
            else:
                decision = "INVESTIGATE"
                reason = "Guardrail violations detected, requires investigation"
                confidence = "MEDIUM"
        else:
            # Analyze primary metrics
            accuracy_analysis = statistical_results.get("accuracy", {})
            auc_analysis = statistical_results.get("auc", {})
            
            significant_improvements = 0
            significant_degradations = 0
            
            for metric_name, metric_data in statistical_results.items():
                if metric_data.get("is_significant", False):
                    if metric_data.get("improvement", 0) > 0:
                        significant_improvements += 1
                    else:
                        significant_degradations += 1
            
            # Decision logic
            if significant_degradations > 0:
                decision = "REJECT"
                reason = "Significant performance degradations detected"
                confidence = "HIGH"
            elif significant_improvements >= 2:
                decision = "PROMOTE"
                reason = "Multiple significant improvements detected"
                confidence = "HIGH"
            elif significant_improvements == 1:
                decision = "PROMOTE"
                reason = "Significant improvement detected"
                confidence = "MEDIUM"
            else:
                decision = "NEUTRAL"
                reason = "No significant differences detected"
                confidence = "MEDIUM"
        
        return {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supporting_evidence": {
                "significant_improvements": significant_improvements,
                "significant_degradations": significant_degradations,
                "guardrail_violations": guardrail_results["total_violations"],
                "business_impact_positive": business_impact["estimated_accuracy_gain"] > 0
            }
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_confidence_level(self, statistical_results: Dict[str, Any]) -> float:
        """Calculate overall confidence level in results."""
        significant_tests = sum(1 for result in statistical_results.values() 
                              if result.get("is_significant", False))
        total_tests = len(statistical_results)
        
        if total_tests == 0:
            return 0.0
        
        return significant_tests / total_tests
    
    def _generate_next_steps(self, recommendation: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps."""
        decision = recommendation["decision"]
        
        if decision == "PROMOTE":
            return [
                "Deploy challenger model to production",
                "Monitor production metrics closely",
                "Prepare rollback plan if needed",
                "Update model registry and documentation"
            ]
        elif decision == "REJECT":
            return [
                "Investigate performance issues",
                "Retrain model with improvements",
                "Address guardrail violations",
                "Consider alternative approaches"
            ]
        elif decision == "INVESTIGATE":
            return [
                "Extend experiment duration",
                "Investigate guardrail violations",
                "Collect additional data",
                "Consider adjusting traffic allocation"
            ]
        else:  # NEUTRAL
            return [
                "Extend experiment duration",
                "Increase sample size",
                "Consider different metrics",
                "Review experiment design"
            ]
    
    # Helper methods for data storage and retrieval
    def _get_model_info(self, model_uri: str) -> Optional[Dict[str, Any]]:
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
            logger.error(f"Error getting model info for {model_uri}: {e}")
            return None
    
    def _store_experiment_config(self, experiment_id: str, config: Dict[str, Any]):
        """Store experiment configuration."""
        if self.redis_client:
            try:
                key = f"{self.config['redis_config']['key_prefix']}{experiment_id}:config"
                self.redis_client.set(key, json.dumps(config))
            except Exception as e:
                logger.error(f"Error storing experiment config: {e}")
        
        # Also store to file
        os.makedirs("experiments", exist_ok=True)
        with open(f"experiments/{experiment_id}_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def _load_experiment_config(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment configuration."""
        # Try Redis first
        if self.redis_client:
            try:
                key = f"{self.config['redis_config']['key_prefix']}{experiment_id}:config"
                config_str = self.redis_client.get(key)
                if config_str:
                    return json.loads(config_str)
            except Exception as e:
                logger.error(f"Error loading experiment config from Redis: {e}")
        
        # Fallback to file
        try:
            with open(f"experiments/{experiment_id}_config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def _store_analysis_results(self, experiment_id: str, results: Dict[str, Any]):
        """Store analysis results."""
        os.makedirs("experiments", exist_ok=True)
        with open(f"experiments/{experiment_id}_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def _validate_experiment_readiness(self, experiment_config: Dict[str, Any]) -> bool:
        """Validate that experiment is ready to start."""
        # Check model availability
        champion_uri = experiment_config["models"]["champion"]["model_uri"]
        challenger_uri = experiment_config["models"]["challenger"]["model_uri"]
        
        try:
            mlflow.models.get_model_info(champion_uri)
            mlflow.models.get_model_info(challenger_uri)
            return True
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _calculate_experiment_duration(self, experiment_config: Dict[str, Any]) -> float:
        """Calculate experiment duration in hours."""
        start_time = datetime.fromisoformat(experiment_config.get("actual_start_time", 
                                                                 experiment_config["start_time"]))
        end_time = datetime.now(timezone.utc)
        return (end_time - start_time).total_seconds() / 3600
    
    def _calculate_sample_sizes(self, metrics_data: Dict[str, Any]) -> Dict[str, int]:
        """Calculate sample sizes for each variant."""
        sample_sizes = {"champion": 0, "challenger": 0}
        
        if "accuracy" in metrics_data:
            sample_sizes["champion"] = len(metrics_data["accuracy"].get("champion", []))
            sample_sizes["challenger"] = len(metrics_data["accuracy"].get("challenger", []))
        
        return sample_sizes


class TrafficAllocator:
    """Handles traffic allocation for A/B testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allocation_rules = {}
    
    def start_allocation(self, experiment_id: str, experiment_config: Dict[str, Any]):
        """Start traffic allocation for experiment."""
        logger.info(f"Starting traffic allocation for {experiment_id}")
        
        allocation = experiment_config["traffic_allocation"]
        self.allocation_rules[experiment_id] = {
            "champion": allocation.get("champion", 0.9),
            "challenger": allocation.get("challenger", 0.1),
            "active": True
        }
    
    def stop_allocation(self, experiment_id: str):
        """Stop traffic allocation for experiment."""
        logger.info(f"Stopping traffic allocation for {experiment_id}")
        
        if experiment_id in self.allocation_rules:
            self.allocation_rules[experiment_id]["active"] = False
    
    def get_model_assignment(self, experiment_id: str, user_id: str = None) -> str:
        """Get model assignment for a request."""
        if experiment_id not in self.allocation_rules:
            return "champion"
        
        rules = self.allocation_rules[experiment_id]
        if not rules["active"]:
            return "champion"
        
        # Simple hash-based assignment for consistency
        if user_id:
            hash_value = hash(user_id) % 100
        else:
            hash_value = random.randint(0, 99)
        
        challenger_threshold = rules["challenger"] * 100
        
        return "challenger" if hash_value < challenger_threshold else "champion"


class ABTestMetricsCollector:
    """Collects and stores A/B test metrics."""
    
    def __init__(self, config: Dict[str, Any], redis_client: Optional[redis.Redis]):
        self.config = config
        self.redis_client = redis_client
        self.active_experiments = set()
        self.metrics_data = defaultdict(lambda: defaultdict(list))
    
    def initialize_experiment(self, experiment_id: str, experiment_config: Dict[str, Any]):
        """Initialize metrics collection for experiment."""
        logger.info(f"Initializing metrics collection for {experiment_id}")
        self.active_experiments.add(experiment_id)
    
    def start_collection(self, experiment_id: str):
        """Start metrics collection."""
        logger.info(f"Starting metrics collection for {experiment_id}")
    
    def stop_collection(self, experiment_id: str):
        """Stop metrics collection."""
        logger.info(f"Stopping metrics collection for {experiment_id}")
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)
    
    def record_prediction(self, experiment_id: str, variant: str, 
                         prediction_data: Dict[str, Any]):
        """Record a prediction result."""
        if experiment_id not in self.active_experiments:
            return
        
        # Extract metrics from prediction data
        accuracy = prediction_data.get("accuracy", 0)
        latency = prediction_data.get("latency", 0)
        confidence = prediction_data.get("confidence", 0)
        
        # Store metrics
        self.metrics_data[experiment_id]["accuracy"][variant].append(accuracy)
        self.metrics_data[experiment_id]["latency"][variant].append(latency)
        self.metrics_data[experiment_id]["confidence"][variant].append(confidence)
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get metrics data for experiment."""
        if experiment_id in self.metrics_data:
            return dict(self.metrics_data[experiment_id])
        
        # Generate sample data for demonstration
        return self._generate_sample_metrics()
    
    def _generate_sample_metrics(self) -> Dict[str, Any]:
        """Generate sample metrics for demonstration."""
        np.random.seed(42)
        
        return {
            "accuracy": {
                "champion": np.random.normal(0.85, 0.02, 1000).tolist(),
                "challenger": np.random.normal(0.87, 0.02, 100).tolist()
            },
            "auc": {
                "champion": np.random.normal(0.88, 0.01, 1000).tolist(),
                "challenger": np.random.normal(0.89, 0.01, 100).tolist()
            },
            "latency": {
                "champion": np.random.normal(150, 20, 1000).tolist(),
                "challenger": np.random.normal(160, 25, 100).tolist()
            },
            "error_rate": {
                "champion": np.random.uniform(0.01, 0.03, 1000).tolist(),
                "challenger": np.random.uniform(0.01, 0.02, 100).tolist()
            }
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="A/B Testing Framework for Model Updates")
    parser.add_argument("--action", required=True, 
                       choices=["create", "start", "stop", "analyze", "status"],
                       help="Action to perform")
    parser.add_argument("--experiment-id", help="Experiment ID")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--champion-model", help="Champion model URI")
    parser.add_argument("--challenger-model", help="Challenger model URI")
    parser.add_argument("--duration-hours", type=int, help="Experiment duration in hours")
    parser.add_argument("--traffic-split", help="Traffic split JSON (e.g., '{\"champion\": 0.9, \"challenger\": 0.1}')")
    parser.add_argument("--config", default="configs/ab_test_config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = ABTestFramework(args.config)
    
    try:
        if args.action == "create":
            if not all([args.experiment_name, args.champion_model, args.challenger_model]):
                print("Error: experiment-name, champion-model, and challenger-model are required for create action")
                sys.exit(1)
            
            traffic_split = None
            if args.traffic_split:
                traffic_split = json.loads(args.traffic_split)
            
            experiment_id = framework.create_experiment(
                args.champion_model,
                args.challenger_model,
                args.experiment_name,
                args.duration_hours,
                traffic_split
            )
            print(f"Experiment created: {experiment_id}")
        
        elif args.action == "start":
            success = framework.start_experiment(args.experiment_id)
            print(f"Experiment start: {'SUCCESS' if success else 'FAILED'}")
        
        elif args.action == "stop":
            success = framework.stop_experiment(args.experiment_id)
            print(f"Experiment stop: {'SUCCESS' if success else 'FAILED'}")
        
        elif args.action == "analyze":
            results = framework.analyze_results(args.experiment_id)
            print(f"Analysis completed: {results['recommendation']['decision']}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
        
        elif args.action == "status":
            experiment_id = args.experiment_id or framework.experiment_id
            if experiment_id:
                config = framework._load_experiment_config(experiment_id)
                if config:
                    print(f"Experiment {experiment_id}: {config['status']}")
                else:
                    print(f"Experiment {experiment_id} not found")
            else:
                print("No active experiment")
    
    except Exception as e:
        logger.error(f"A/B testing operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()