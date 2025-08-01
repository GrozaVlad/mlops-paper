#!/usr/bin/env python3
"""
A/B Test Preparation Script
Prepares A/B test configuration for model comparison experiments.
"""

import argparse
import json
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import mlflow
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABTestPreparator:
    """Prepares A/B test configurations and validates experiment setup."""
    
    def __init__(self):
        """Initialize A/B test preparator."""
        self.mlflow_client = mlflow.tracking.MlflowClient()
    
    def prepare_ab_test(self, champion_model_uri: str, challenger_model_uri: str,
                       traffic_split: float = 0.1, duration_hours: int = 72,
                       experiment_name: str = None) -> Dict[str, Any]:
        """Prepare A/B test configuration.
        
        Args:
            champion_model_uri: URI of current production model
            challenger_model_uri: URI of new model to test
            traffic_split: Percentage of traffic to send to challenger (0.0-1.0)
            duration_hours: Duration of the experiment
            experiment_name: Name for the experiment
            
        Returns:
            A/B test configuration
        """
        logger.info("Preparing A/B test configuration...")
        
        # Generate experiment name if not provided
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"model_ab_test_{timestamp}"
        
        # Validate models
        champion_info = self._get_model_info(champion_model_uri)
        challenger_info = self._get_model_info(challenger_model_uri)
        
        if not champion_info:
            raise ValueError(f"Champion model not found: {champion_model_uri}")
        if not challenger_info:
            raise ValueError(f"Challenger model not found: {challenger_model_uri}")
        
        # Validate traffic split
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        
        # Calculate sample size requirements
        sample_size_analysis = self._calculate_sample_size_requirements(traffic_split, duration_hours)
        
        # Generate experiment configuration
        experiment_config = {
            "experiment_info": {
                "experiment_name": experiment_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "duration_hours": duration_hours,
                "estimated_end_time": (datetime.now(timezone.utc) + timedelta(hours=duration_hours)).isoformat()
            },
            "models": {
                "champion": {
                    "model_uri": champion_model_uri,
                    "model_info": champion_info,
                    "traffic_allocation": 1.0 - traffic_split,
                    "role": "baseline"
                },
                "challenger": {
                    "model_uri": challenger_model_uri,
                    "model_info": challenger_info,
                    "traffic_allocation": traffic_split,
                    "role": "variant"
                }
            },
            "traffic_configuration": {
                "allocation_strategy": "hash_based",
                "champion_percentage": (1.0 - traffic_split) * 100,
                "challenger_percentage": traffic_split * 100,
                "ramp_up_strategy": "immediate",  # or "gradual"
                "sticky_sessions": True
            },
            "metrics_configuration": {
                "primary_metrics": [
                    "accuracy",
                    "auc_score",
                    "precision",
                    "recall",
                    "f1_score"
                ],
                "secondary_metrics": [
                    "prediction_latency",
                    "throughput_rps",
                    "error_rate",
                    "prediction_confidence"
                ],
                "business_metrics": [
                    "user_satisfaction",
                    "prediction_quality",
                    "model_reliability"
                ],
                "collection_interval_seconds": 60,
                "aggregation_window_minutes": 5
            },
            "statistical_configuration": {
                "significance_level": 0.05,
                "statistical_power": 0.8,
                "minimum_effect_size": 0.02,  # 2% improvement
                "multiple_testing_correction": "bonferroni",
                "confidence_interval": 0.95
            },
            "sample_size_analysis": sample_size_analysis,
            "guardrail_configuration": {
                "max_error_rate": 0.05,
                "max_latency_increase_pct": 20,
                "min_accuracy_threshold": 0.75,
                "max_confidence_drop_pct": 10,
                "violation_actions": ["alert", "reduce_traffic", "stop_experiment"]
            },
            "early_stopping_rules": {
                "enable_early_stopping": True,
                "minimum_runtime_hours": 12,
                "check_interval_hours": 6,
                "early_stop_conditions": {
                    "significance_achieved": True,
                    "guardrail_violation": True,
                    "sufficient_sample_size": True,
                    "business_impact_threshold": True
                }
            },
            "deployment_configuration": {
                "kubernetes_config": {
                    "namespace": "drugban-ab-test",
                    "champion_deployment": "drugban-api-champion",
                    "challenger_deployment": "drugban-api-challenger",
                    "load_balancer_config": {
                        "service_name": "drugban-api-ab-test",
                        "algorithm": "hash_based",
                        "session_affinity": "ClientIP"
                    }
                },
                "model_serving_config": {
                    "champion_endpoint": "http://champion-service:8000/predict",
                    "challenger_endpoint": "http://challenger-service:8000/predict",
                    "timeout_seconds": 30,
                    "retry_attempts": 2
                }
            },
            "monitoring_configuration": {
                "prometheus_config": {
                    "metrics_endpoint": "/metrics",
                    "scrape_interval": "15s",
                    "custom_metrics": [
                        "ab_test_prediction_total",
                        "ab_test_prediction_duration",
                        "ab_test_prediction_accuracy",
                        "ab_test_model_assignment_total"
                    ]
                },
                "grafana_dashboard": {
                    "dashboard_name": f"AB Test - {experiment_name}",
                    "panels": [
                        "traffic_distribution",
                        "model_performance_comparison",
                        "statistical_significance",
                        "guardrail_metrics"
                    ]
                },
                "alerting_rules": [
                    {
                        "name": "high_error_rate",
                        "condition": "error_rate > 0.05",
                        "action": "send_alert"
                    },
                    {
                        "name": "significant_performance_drop",
                        "condition": "accuracy_drop > 0.05",
                        "action": "reduce_challenger_traffic"
                    }
                ]
            },
            "validation_checklist": self._generate_validation_checklist(champion_info, challenger_info),
            "rollback_plan": {
                "automatic_rollback": True,
                "rollback_triggers": [
                    "guardrail_violation",
                    "high_error_rate",
                    "significant_performance_degradation"
                ],
                "rollback_steps": [
                    "stop_challenger_traffic",
                    "verify_champion_health",
                    "notify_team",
                    "create_incident_report"
                ]
            }
        }
        
        # Add experiment readiness assessment
        readiness_assessment = self._assess_experiment_readiness(experiment_config)
        experiment_config["readiness_assessment"] = readiness_assessment
        
        logger.info(f"A/B test configuration prepared: {experiment_name}")
        logger.info(f"Traffic split: {traffic_split*100:.1f}% to challenger")
        logger.info(f"Estimated sample sizes: Champion={sample_size_analysis['champion_samples']}, "
                   f"Challenger={sample_size_analysis['challenger_samples']}")
        
        return experiment_config
    
    def _get_model_info(self, model_uri: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        try:
            # Get model info from MLflow
            model_info = mlflow.models.get_model_info(model_uri)
            
            # Get run information
            run = self.mlflow_client.get_run(model_info.run_id)
            
            # Extract model metrics if available
            metrics = {}
            for key, value in run.data.metrics.items():
                if key in ["accuracy", "auc", "precision", "recall", "f1_score"]:
                    metrics[key] = value
            
            return {
                "model_uri": model_uri,
                "run_id": model_info.run_id,
                "model_uuid": model_info.model_uuid,
                "creation_timestamp": model_info.utc_time_created,
                "model_size_bytes": model_info.model_size_bytes if hasattr(model_info, 'model_size_bytes') else None,
                "flavors": list(model_info.flavors.keys()),
                "training_metrics": metrics,
                "run_name": run.info.run_name,
                "experiment_id": run.info.experiment_id,
                "tags": run.data.tags,
                "parameters": run.data.params
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_uri}: {e}")
            return None
    
    def _calculate_sample_size_requirements(self, traffic_split: float, 
                                          duration_hours: int) -> Dict[str, Any]:
        """Calculate required sample sizes for statistical significance."""
        # Statistical parameters
        alpha = 0.05  # significance level
        beta = 0.2   # type II error (power = 1 - beta = 0.8)
        effect_size = 0.02  # minimum detectable effect (2% improvement)
        
        # Estimate traffic rate (requests per hour)
        # This would typically come from historical data
        estimated_rph = 1000  # requests per hour
        
        # Calculate total expected requests
        total_requests = estimated_rph * duration_hours
        challenger_requests = int(total_requests * traffic_split)
        champion_requests = int(total_requests * (1 - traffic_split))
        
        # Calculate minimum sample size for statistical power
        # Using simplified formula for two-proportion z-test
        z_alpha = 1.96  # z-score for alpha=0.05 (two-tailed)
        z_beta = 0.84   # z-score for beta=0.2 (power=0.8)
        
        p1 = 0.85  # assumed baseline accuracy
        p2 = p1 + effect_size  # expected improvement
        p_pooled = (p1 + p2) / 2
        
        n_required = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (effect_size**2)
        n_required = int(np.ceil(n_required))
        
        # Check if expected samples meet requirements
        champion_adequate = champion_requests >= n_required
        challenger_adequate = challenger_requests >= n_required
        
        return {
            "statistical_parameters": {
                "significance_level": alpha,
                "statistical_power": 1 - beta,
                "minimum_effect_size": effect_size,
                "baseline_accuracy": p1,
                "expected_improvement": effect_size
            },
            "sample_size_calculation": {
                "required_samples_per_group": n_required,
                "total_required_samples": n_required * 2
            },
            "expected_samples": {
                "champion_samples": champion_requests,
                "challenger_samples": challenger_requests,
                "total_samples": total_requests
            },
            "adequacy_assessment": {
                "champion_adequate": champion_adequate,
                "challenger_adequate": challenger_adequate,
                "overall_adequate": champion_adequate and challenger_adequate,
                "champion_coverage_pct": (champion_requests / n_required) * 100 if n_required > 0 else 0,
                "challenger_coverage_pct": (challenger_requests / n_required) * 100 if n_required > 0 else 0
            },
            "recommendations": self._generate_sample_size_recommendations(
                champion_adequate, challenger_adequate, traffic_split, duration_hours
            )
        }
    
    def _generate_sample_size_recommendations(self, champion_adequate: bool, 
                                            challenger_adequate: bool,
                                            traffic_split: float, 
                                            duration_hours: int) -> List[str]:
        """Generate recommendations based on sample size analysis."""
        recommendations = []
        
        if not champion_adequate:
            recommendations.append(
                "Champion sample size may be insufficient for statistical significance"
            )
        
        if not challenger_adequate:
            recommendations.append(
                "Challenger sample size may be insufficient for statistical significance"
            )
            if traffic_split < 0.2:
                recommendations.append(
                    f"Consider increasing challenger traffic split from {traffic_split*100:.1f}% to 20%"
                )
        
        if not (champion_adequate and challenger_adequate):
            if duration_hours < 168:  # less than 1 week
                recommendations.append(
                    f"Consider extending experiment duration beyond {duration_hours} hours"
                )
            recommendations.append(
                "Monitor sample sizes during experiment and extend if needed"
            )
        
        if champion_adequate and challenger_adequate:
            recommendations.append(
                "Sample sizes are adequate for detecting statistical significance"
            )
        
        return recommendations
    
    def _generate_validation_checklist(self, champion_info: Dict[str, Any], 
                                     challenger_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate pre-experiment validation checklist."""
        checklist = [
            {
                "item": "Model Compatibility Check",
                "description": "Verify both models use compatible input/output schemas",
                "required": True,
                "validation_method": "schema_comparison",
                "status": "pending"
            },
            {
                "item": "Performance Baseline",
                "description": "Ensure champion model has established performance baseline",
                "required": True,
                "validation_method": "metrics_availability",
                "status": "pending"
            },
            {
                "item": "Model Deployment",
                "description": "Verify both models can be deployed to test environment",
                "required": True,
                "validation_method": "deployment_test",
                "status": "pending"
            },
            {
                "item": "Traffic Routing",
                "description": "Validate traffic routing and load balancer configuration",
                "required": True,
                "validation_method": "routing_test",
                "status": "pending"
            },
            {
                "item": "Metrics Collection",
                "description": "Ensure metrics collection is properly configured",
                "required": True,
                "validation_method": "metrics_test",
                "status": "pending"
            },
            {
                "item": "Alerting Setup",
                "description": "Configure alerts for guardrail violations",
                "required": True,
                "validation_method": "alert_test",
                "status": "pending"
            },
            {
                "item": "Rollback Mechanism",
                "description": "Verify rollback procedures work correctly",
                "required": True,
                "validation_method": "rollback_test",
                "status": "pending"
            },
            {
                "item": "Team Notification",
                "description": "Notify relevant teams about the experiment",
                "required": False,
                "validation_method": "manual_check",
                "status": "pending"
            }
        ]
        
        return checklist
    
    def _assess_experiment_readiness(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall readiness for running the experiment."""
        sample_analysis = experiment_config["sample_size_analysis"]
        
        # Check critical readiness factors
        readiness_factors = {
            "models_available": True,  # Already validated
            "sample_size_adequate": sample_analysis["adequacy_assessment"]["overall_adequate"],
            "configuration_complete": True,  # Config is generated
            "monitoring_configured": True,  # Config includes monitoring
            "guardrails_defined": True   # Guardrails are configured
        }
        
        # Calculate overall readiness score
        ready_count = sum(readiness_factors.values())
        total_factors = len(readiness_factors)
        readiness_score = ready_count / total_factors
        
        # Determine readiness status
        if readiness_score >= 0.9:
            status = "READY"
        elif readiness_score >= 0.7:
            status = "NEEDS_ATTENTION"
        else:
            status = "NOT_READY"
        
        # Generate readiness recommendations
        recommendations = []
        if not readiness_factors["sample_size_adequate"]:
            recommendations.append("Increase experiment duration or traffic allocation for adequate sample size")
        
        if status == "READY":
            recommendations.append("Experiment is ready to start")
        elif status == "NEEDS_ATTENTION":
            recommendations.append("Address minor issues before starting experiment")
        else:
            recommendations.append("Resolve critical issues before proceeding")
        
        return {
            "status": status,
            "readiness_score": readiness_score,
            "factors": readiness_factors,
            "ready_factors": ready_count,
            "total_factors": total_factors,
            "recommendations": recommendations,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate experiment configuration for completeness and correctness.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        logger.info("Validating experiment configuration...")
        
        errors = []
        
        # Required sections
        required_sections = [
            "experiment_info",
            "models", 
            "traffic_configuration",
            "metrics_configuration",
            "statistical_configuration"
        ]
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Model validation
        if "models" in config:
            if "champion" not in config["models"]:
                errors.append("Champion model configuration missing")
            if "challenger" not in config["models"]:
                errors.append("Challenger model configuration missing")
            
            # Check model URIs
            for model_type in ["champion", "challenger"]:
                if model_type in config["models"]:
                    if "model_uri" not in config["models"][model_type]:
                        errors.append(f"{model_type} model URI missing")
        
        # Traffic configuration validation
        if "traffic_configuration" in config:
            champion_pct = config["traffic_configuration"].get("champion_percentage", 0)
            challenger_pct = config["traffic_configuration"].get("challenger_percentage", 0)
            
            if abs(champion_pct + challenger_pct - 100) > 0.01:
                errors.append("Traffic percentages do not sum to 100%")
            
            if challenger_pct < 1 or challenger_pct > 50:
                errors.append("Challenger traffic percentage should be between 1% and 50%")
        
        # Statistical configuration validation
        if "statistical_configuration" in config:
            stat_config = config["statistical_configuration"]
            
            if stat_config.get("significance_level", 0) <= 0 or stat_config.get("significance_level", 1) >= 1:
                errors.append("Significance level must be between 0 and 1")
            
            if stat_config.get("statistical_power", 0) <= 0 or stat_config.get("statistical_power", 1) >= 1:
                errors.append("Statistical power must be between 0 and 1")
        
        # Sample size validation
        if "sample_size_analysis" in config:
            adequacy = config["sample_size_analysis"]["adequacy_assessment"]
            if not adequacy.get("overall_adequate", False):
                errors.append("Insufficient sample size for statistical significance")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Experiment configuration validation passed")
        else:
            logger.warning(f"Experiment configuration validation failed with {len(errors)} errors")
        
        return is_valid, errors


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare A/B Test Configuration")
    parser.add_argument("--champion-model", required=True, help="Champion model URI")
    parser.add_argument("--challenger-model", required=True, help="Challenger model URI")
    parser.add_argument("--traffic-split", type=float, default=0.1, 
                       help="Traffic percentage for challenger (0.0-1.0)")
    parser.add_argument("--duration-hours", type=int, default=72, 
                       help="Experiment duration in hours")
    parser.add_argument("--experiment-name", help="Experiment name")
    parser.add_argument("--output", required=True, help="Output file for configuration")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration without creating it")
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = ABTestPreparator()
    
    try:
        if args.validate_only and Path(args.output).exists():
            # Load and validate existing configuration
            with open(args.output, 'r') as f:
                config = json.load(f)
            
            is_valid, errors = preparator.validate_experiment_config(config)
            
            if is_valid:
                print("✅ Configuration validation passed")
                sys.exit(0)
            else:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
        
        # Prepare new configuration
        config = preparator.prepare_ab_test(
            args.champion_model,
            args.challenger_model,
            args.traffic_split,
            args.duration_hours,
            args.experiment_name
        )
        
        # Validate the configuration
        is_valid, errors = preparator.validate_experiment_config(config)
        
        if not is_valid:
            print("❌ Generated configuration has validation errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        # Save configuration
        os.makedirs(Path(args.output).parent, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Print summary
        print(f"✅ A/B test configuration prepared: {config['experiment_info']['experiment_name']}")
        print(f"📊 Traffic split: {args.traffic_split*100:.1f}% to challenger")
        print(f"⏱️  Duration: {args.duration_hours} hours")
        print(f"📁 Configuration saved to: {args.output}")
        
        # Print readiness assessment
        readiness = config["readiness_assessment"]
        print(f"🚦 Readiness: {readiness['status']} ({readiness['readiness_score']:.1%})")
        
        if readiness["recommendations"]:
            print("💡 Recommendations:")
            for rec in readiness["recommendations"]:
                print(f"  - {rec}")
        
    except Exception as e:
        logger.error(f"A/B test preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()