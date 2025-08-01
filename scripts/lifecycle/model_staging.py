#!/usr/bin/env python3
"""
Model Staging and Lifecycle Management
Comprehensive model staging system for managing model lifecycle from development to production.
"""

import argparse
import json
import sys
import time
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class DeploymentStatus(Enum):
    """Deployment status for staged models."""
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    FAILED = "FAILED"
    ROLLING_BACK = "ROLLING_BACK"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelInfo:
    """Model information structure."""
    name: str
    version: str
    stage: str
    run_id: str
    model_uri: str
    creation_timestamp: str
    current_stage: str
    description: str
    tags: Dict[str, str]
    latest_versions: Dict[str, str]
    performance_metrics: Dict[str, float]
    deployment_status: str
    deployment_config: Dict[str, Any]


@dataclass
class StageTransition:
    """Stage transition information."""
    model_name: str
    from_stage: str
    to_stage: str
    version: str
    transition_timestamp: str
    requested_by: str
    approval_status: str
    approval_timestamp: Optional[str]
    transition_reason: str
    validation_results: Dict[str, Any]
    rollback_info: Optional[Dict[str, Any]]


class ModelStagingManager:
    """Manages model staging and lifecycle operations."""
    
    def __init__(self, config_path: str = "configs/model_staging_config.yaml"):
        """Initialize model staging manager.
        
        Args:
            config_path: Path to staging configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_client = MlflowClient()
        
        # Transition tracking
        self.transition_history = []
        self.active_transitions = {}
        
        # Model validation
        self.validator = ModelValidator(self.config)
        self.deployment_manager = DeploymentManager(self.config)
        
        # Setup MLflow tracking
        self._setup_mlflow()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load staging configuration."""
        default_config = {
            "staging_rules": {
                "require_approval": {
                    "staging": False,
                    "production": True,
                    "archived": False
                },
                "validation_required": {
                    "staging": True,
                    "production": True,
                    "archived": False
                },
                "performance_thresholds": {
                    "staging": {
                        "min_accuracy": 0.70,
                        "min_auc": 0.75,
                        "max_inference_time_ms": 2000
                    },
                    "production": {
                        "min_accuracy": 0.85,
                        "min_auc": 0.90,
                        "max_inference_time_ms": 1000
                    }
                }
            },
            "deployment_config": {
                "staging": {
                    "environment": "staging",
                    "replicas": 1,
                    "resources": {
                        "cpu": "500m",
                        "memory": "1Gi"
                    },
                    "auto_deploy": True
                },
                "production": {
                    "environment": "production",
                    "replicas": 3,
                    "resources": {
                        "cpu": "1000m",
                        "memory": "2Gi"
                    },
                    "auto_deploy": False,
                    "blue_green": True
                }
            },
            "validation_config": {
                "test_datasets": {
                    "staging": "data/validation/",
                    "production": "data/test/"
                },
                "validation_metrics": [
                    "accuracy", "auc", "precision", "recall", "f1_score"
                ],
                "benchmark_comparison": True,
                "performance_regression_threshold": 0.05
            },
            "approval_workflow": {
                "production_approvers": ["model-admin", "ml-ops-lead"],
                "approval_timeout_hours": 72,
                "automatic_rollback": True,
                "rollback_conditions": {
                    "performance_degradation": 0.10,
                    "error_rate_increase": 0.05,
                    "latency_increase": 0.50
                }
            },
            "archival_policy": {
                "auto_archive_after_days": 90,
                "keep_n_versions": 5,
                "archive_performance_threshold": 0.60
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
    
    def _setup_mlflow(self):
        """Setup MLflow tracking configuration."""
        try:
            # Ensure experiments exist
            experiments = ["model_staging", "model_validation", "model_deployment"]
            for exp_name in experiments:
                try:
                    mlflow.get_experiment_by_name(exp_name)
                except:
                    mlflow.create_experiment(exp_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    def list_models(self, stage: Optional[str] = None) -> List[ModelInfo]:
        """List models in specified stage.
        
        Args:
            stage: Filter by stage (None for all)
            
        Returns:
            List of model information
        """
        logger.info(f"Listing models" + (f" in stage: {stage}" if stage else ""))
        
        models = []
        
        try:
            # Get all registered models
            registered_models = self.mlflow_client.search_registered_models()
            
            for rm in registered_models:
                model_versions = self.mlflow_client.search_model_versions(
                    f"name='{rm.name}'"
                )
                
                # Get latest versions by stage
                latest_versions = {}
                for mv in model_versions:
                    if mv.current_stage not in latest_versions:
                        latest_versions[mv.current_stage] = mv.version
                    elif int(mv.version) > int(latest_versions[mv.current_stage]):
                        latest_versions[mv.current_stage] = mv.version
                
                # Filter by stage if specified
                versions_to_include = model_versions
                if stage:
                    versions_to_include = [mv for mv in model_versions if mv.current_stage == stage]
                
                for mv in versions_to_include:
                    # Get performance metrics
                    performance_metrics = self._get_model_performance(mv.run_id)
                    
                    # Get deployment status
                    deployment_status = self._get_deployment_status(rm.name, mv.version, mv.current_stage)
                    
                    model_info = ModelInfo(
                        name=rm.name,
                        version=mv.version,
                        stage=mv.current_stage,
                        run_id=mv.run_id,
                        model_uri=f"models:/{rm.name}/{mv.version}",
                        creation_timestamp=mv.creation_timestamp,
                        current_stage=mv.current_stage,
                        description=mv.description or "",
                        tags=mv.tags or {},
                        latest_versions=latest_versions,
                        performance_metrics=performance_metrics,
                        deployment_status=deployment_status.value,
                        deployment_config=self._get_deployment_config(mv.current_stage)
                    )
                    
                    models.append(model_info)
                    
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
        
        logger.info(f"Found {len(models)} models")
        return models
    
    def transition_model_stage(self, model_name: str, version: str, 
                             target_stage: str, description: str = "",
                             force: bool = False) -> bool:
        """Transition model to target stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            target_stage: Target stage (Staging, Production, Archived)
            description: Transition description
            force: Force transition without validation
            
        Returns:
            Success status
        """
        logger.info(f"Transitioning model {model_name} v{version} to {target_stage}")
        
        try:
            # Validate target stage
            if target_stage not in [stage.value for stage in ModelStage]:
                raise ValueError(f"Invalid target stage: {target_stage}")
            
            # Get current model version
            current_mv = self.mlflow_client.get_model_version(model_name, version)
            current_stage = current_mv.current_stage
            
            if current_stage == target_stage:
                logger.info(f"Model already in {target_stage} stage")
                return True
            
            # Create transition record
            transition = StageTransition(
                model_name=model_name,
                from_stage=current_stage,
                to_stage=target_stage,
                version=version,
                transition_timestamp=datetime.now(timezone.utc).isoformat(),
                requested_by=os.getenv("USER", "system"),
                approval_status="PENDING",
                approval_timestamp=None,
                transition_reason=description,
                validation_results={},
                rollback_info=None
            )
            
            # Validation phase
            if not force and self.config["staging_rules"]["validation_required"].get(target_stage.lower(), False):
                logger.info("Running model validation...")
                validation_results = self.validator.validate_model_for_stage(
                    model_name, version, target_stage
                )
                
                transition.validation_results = validation_results
                
                if not validation_results.get("passed", False):
                    logger.error(f"Model validation failed: {validation_results.get('errors', [])}")
                    transition.approval_status = "REJECTED"
                    self.transition_history.append(transition)
                    return False
            
            # Approval phase
            if not force and self.config["staging_rules"]["require_approval"].get(target_stage.lower(), False):
                logger.info("Approval required for this transition")
                
                # For production transitions, require manual approval
                if target_stage == ModelStage.PRODUCTION.value:
                    approval_result = self._request_approval(transition)
                    if not approval_result:
                        logger.warning("Approval not granted, transition pending")
                        self.active_transitions[f"{model_name}_{version}"] = transition
                        return False
                
                transition.approval_status = "APPROVED"
                transition.approval_timestamp = datetime.now(timezone.utc).isoformat()
            else:
                transition.approval_status = "AUTO_APPROVED"
                transition.approval_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Execute transition
            success = self._execute_stage_transition(transition)
            
            if success:
                # Update MLflow model stage
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=target_stage,
                    description=description
                )
                
                # Deploy if required
                if self._should_auto_deploy(target_stage):
                    deployment_success = self.deployment_manager.deploy_model(
                        model_name, version, target_stage
                    )
                    if not deployment_success:
                        logger.warning("Model staged but deployment failed")
                
                transition.approval_status = "COMPLETED"
                logger.info(f"Model transition completed successfully")
            else:
                transition.approval_status = "FAILED"
                logger.error("Model transition failed")
            
            self.transition_history.append(transition)
            return success
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            return False
    
    def promote_model(self, model_name: str, from_stage: str, to_stage: str,
                     performance_comparison: bool = True) -> bool:
        """Promote model from one stage to another.
        
        Args:
            model_name: Name of the model
            from_stage: Source stage
            to_stage: Target stage
            performance_comparison: Compare with current model in target stage
            
        Returns:
            Success status
        """
        logger.info(f"Promoting model {model_name} from {from_stage} to {to_stage}")
        
        try:
            # Get latest version in source stage
            source_versions = self.mlflow_client.search_model_versions(
                f"name='{model_name}' and current_stage='{from_stage}'"
            )
            
            if not source_versions:
                logger.error(f"No model found in {from_stage} stage")
                return False
            
            # Get latest version (highest version number)
            latest_version = max(source_versions, key=lambda x: int(x.version))
            
            # Performance comparison
            if performance_comparison and to_stage == ModelStage.PRODUCTION.value:
                comparison_result = self._compare_with_production(model_name, latest_version.version)
                if not comparison_result.get("promote", False):
                    logger.warning(f"Performance comparison failed: {comparison_result.get('reason')}")
                    return False
            
            # Execute promotion
            return self.transition_model_stage(
                model_name=model_name,
                version=latest_version.version,
                target_stage=to_stage,
                description=f"Promoted from {from_stage}"
            )
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def archive_model(self, model_name: str, version: str, reason: str = "") -> bool:
        """Archive a model version.
        
        Args:
            model_name: Name of the model
            version: Model version
            reason: Archival reason
            
        Returns:
            Success status
        """
        logger.info(f"Archiving model {model_name} v{version}")
        
        try:
            # Check if model is currently in production
            current_mv = self.mlflow_client.get_model_version(model_name, version)
            if current_mv.current_stage == ModelStage.PRODUCTION.value:
                logger.warning("Cannot archive production model without replacement")
                
                # Check for other production models
                prod_versions = self.mlflow_client.search_model_versions(
                    f"name='{model_name}' and current_stage='Production'"
                )
                
                if len(prod_versions) <= 1:
                    logger.error("Cannot archive last production model")
                    return False
            
            # Archive the model
            success = self.transition_model_stage(
                model_name=model_name,
                version=version,
                target_stage=ModelStage.ARCHIVED.value,
                description=f"Archived: {reason}"
            )
            
            if success:
                # Update deployment status
                self._update_deployment_status(model_name, version, DeploymentStatus.ARCHIVED)
                
                # Clean up deployments
                self.deployment_manager.cleanup_deployment(model_name, version)
            
            return success
            
        except Exception as e:
            logger.error(f"Error archiving model: {e}")
            return False
    
    def auto_archive_old_models(self) -> int:
        """Archive old models based on policy.
        
        Returns:
            Number of models archived
        """
        logger.info("Running automatic model archival...")
        
        archived_count = 0
        policy = self.config["archival_policy"]
        
        try:
            # Get all registered models
            registered_models = self.mlflow_client.search_registered_models()
            
            for rm in registered_models:
                model_versions = self.mlflow_client.search_model_versions(
                    f"name='{rm.name}'"
                )
                
                # Group by stage
                versions_by_stage = {}
                for mv in model_versions:
                    stage = mv.current_stage
                    if stage not in versions_by_stage:
                        versions_by_stage[stage] = []
                    versions_by_stage[stage].append(mv)
                
                # Archive old versions
                for stage, versions in versions_by_stage.items():
                    if stage in [ModelStage.ARCHIVED.value, ModelStage.PRODUCTION.value]:
                        continue  # Skip already archived and production models
                    
                    # Sort by creation timestamp
                    versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
                    
                    # Keep only the latest N versions
                    keep_count = policy["keep_n_versions"]
                    if len(versions) > keep_count:
                        to_archive = versions[keep_count:]
                        
                        for mv in to_archive:
                            # Check age
                            creation_date = datetime.fromisoformat(mv.creation_timestamp.replace('Z', '+00:00'))
                            age_days = (datetime.now(timezone.utc) - creation_date).days
                            
                            if age_days >= policy["auto_archive_after_days"]:
                                # Check performance threshold
                                performance = self._get_model_performance(mv.run_id)
                                accuracy = performance.get("accuracy", 0)
                                
                                if accuracy < policy["archive_performance_threshold"]:
                                    success = self.archive_model(
                                        rm.name, mv.version, 
                                        f"Auto-archived: old model with low performance ({accuracy:.3f})"
                                    )
                                    if success:
                                        archived_count += 1
                                        logger.info(f"Auto-archived {rm.name} v{mv.version}")
        
        except Exception as e:
            logger.error(f"Error in auto archival: {e}")
        
        logger.info(f"Auto-archived {archived_count} models")
        return archived_count
    
    def rollback_model(self, model_name: str, target_stage: str = "Production") -> bool:
        """Rollback to previous model version in stage.
        
        Args:
            model_name: Name of the model
            target_stage: Stage to rollback in
            
        Returns:
            Success status
        """
        logger.info(f"Rolling back model {model_name} in {target_stage}")
        
        try:
            # Get model versions in target stage
            stage_versions = self.mlflow_client.search_model_versions(
                f"name='{model_name}' and current_stage='{target_stage}'"
            )
            
            if len(stage_versions) < 2:
                logger.error("No previous version available for rollback")
                return False
            
            # Sort by version number
            stage_versions.sort(key=lambda x: int(x.version), reverse=True)
            current_version = stage_versions[0]
            previous_version = stage_versions[1]
            
            logger.info(f"Rolling back from v{current_version.version} to v{previous_version.version}")
            
            # Archive current version
            archive_success = self.transition_model_stage(
                model_name=model_name,
                version=current_version.version,
                target_stage=ModelStage.ARCHIVED.value,
                description="Rolled back due to issues",
                force=True
            )
            
            if not archive_success:
                logger.error("Failed to archive current version")
                return False
            
            # Deploy previous version
            deployment_success = self.deployment_manager.deploy_model(
                model_name, previous_version.version, target_stage
            )
            
            if deployment_success:
                logger.info("Rollback completed successfully")
                return True
            else:
                logger.error("Rollback deployment failed")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            return False
    
    def get_stage_transition_history(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get stage transition history.
        
        Args:
            model_name: Filter by model name (None for all)
            
        Returns:
            List of transition records
        """
        transitions = self.transition_history
        
        if model_name:
            transitions = [t for t in transitions if t.model_name == model_name]
        
        return [asdict(t) for t in transitions]
    
    def get_deployment_status(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get detailed deployment status.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Deployment status information
        """
        try:
            mv = self.mlflow_client.get_model_version(model_name, version)
            deployment_status = self._get_deployment_status(model_name, version, mv.current_stage)
            
            return {
                "model_name": model_name,
                "version": version,
                "stage": mv.current_stage,
                "deployment_status": deployment_status.value,
                "deployment_details": self.deployment_manager.get_deployment_info(model_name, version),
                "health_status": self.deployment_manager.check_health(model_name, version),
                "performance_metrics": self._get_model_performance(mv.run_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _get_model_performance(self, run_id: str) -> Dict[str, float]:
        """Get model performance metrics from MLflow run."""
        try:
            run = self.mlflow_client.get_run(run_id)
            metrics = run.data.metrics
            
            # Extract relevant metrics
            performance_metrics = {}
            for metric_name in ["accuracy", "auc", "precision", "recall", "f1_score"]:
                if metric_name in metrics:
                    performance_metrics[metric_name] = metrics[metric_name]
            
            return performance_metrics
            
        except Exception as e:
            logger.warning(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_deployment_status(self, model_name: str, version: str, stage: str) -> DeploymentStatus:
        """Get current deployment status."""
        try:
            deployment_info = self.deployment_manager.get_deployment_info(model_name, version)
            if deployment_info:
                return DeploymentStatus(deployment_info.get("status", "UNKNOWN"))
            else:
                return DeploymentStatus.PENDING
        except:
            return DeploymentStatus.PENDING
    
    def _get_deployment_config(self, stage: str) -> Dict[str, Any]:
        """Get deployment configuration for stage."""
        return self.config["deployment_config"].get(stage.lower(), {})
    
    def _should_auto_deploy(self, stage: str) -> bool:
        """Check if stage should auto-deploy."""
        deployment_config = self._get_deployment_config(stage)
        return deployment_config.get("auto_deploy", False)
    
    def _execute_stage_transition(self, transition: StageTransition) -> bool:
        """Execute the actual stage transition."""
        try:
            # Log transition to MLflow
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("model_staging").experiment_id):
                mlflow.log_param("model_name", transition.model_name)
                mlflow.log_param("version", transition.version)
                mlflow.log_param("from_stage", transition.from_stage)
                mlflow.log_param("to_stage", transition.to_stage)
                mlflow.log_param("requested_by", transition.requested_by)
                
                if transition.validation_results:
                    for key, value in transition.validation_results.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"validation_{key}", value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing transition: {e}")
            return False
    
    def _request_approval(self, transition: StageTransition) -> bool:
        """Request approval for transition (simplified implementation)."""
        # In a real implementation, this would integrate with approval systems
        # For demo purposes, we'll auto-approve
        logger.info(f"Approval requested for {transition.model_name} v{transition.version}")
        return True
    
    def _compare_with_production(self, model_name: str, candidate_version: str) -> Dict[str, Any]:
        """Compare candidate model with current production model."""
        try:
            # Get current production version
            prod_versions = self.mlflow_client.search_model_versions(
                f"name='{model_name}' and current_stage='Production'"
            )
            
            if not prod_versions:
                return {"promote": True, "reason": "No current production model"}
            
            current_prod = max(prod_versions, key=lambda x: int(x.version))
            
            # Get performance metrics
            candidate_metrics = self._get_model_performance(
                self.mlflow_client.get_model_version(model_name, candidate_version).run_id
            )
            current_metrics = self._get_model_performance(current_prod.run_id)
            
            # Compare key metrics
            threshold = self.config["validation_config"]["performance_regression_threshold"]
            
            for metric in ["accuracy", "auc"]:
                if metric in candidate_metrics and metric in current_metrics:
                    improvement = candidate_metrics[metric] - current_metrics[metric]
                    if improvement < -threshold:  # Performance regression
                        return {
                            "promote": False,
                            "reason": f"Performance regression in {metric}: {improvement:.3f}"
                        }
            
            return {"promote": True, "reason": "Performance comparison passed"}
            
        except Exception as e:
            logger.warning(f"Error in performance comparison: {e}")
            return {"promote": True, "reason": "Comparison failed, allowing promotion"}
    
    def _update_deployment_status(self, model_name: str, version: str, status: DeploymentStatus):
        """Update deployment status (simplified implementation)."""
        # In a real implementation, this would update a deployment database
        logger.info(f"Updated deployment status for {model_name} v{version}: {status.value}")


class ModelValidator:
    """Validates models for stage transitions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def validate_model_for_stage(self, model_name: str, version: str, target_stage: str) -> Dict[str, Any]:
        """Validate model for target stage.
        
        Args:
            model_name: Name of the model
            version: Model version  
            target_stage: Target stage
            
        Returns:
            Validation results
        """
        logger.info(f"Validating model {model_name} v{version} for {target_stage}")
        
        validation_results = {
            "passed": False,
            "score": 0.0,
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Get model performance thresholds for target stage
            thresholds = self.config["staging_rules"]["performance_thresholds"].get(
                target_stage.lower(), {}
            )
            
            if not thresholds:
                validation_results["passed"] = True
                validation_results["warnings"].append(f"No thresholds defined for {target_stage}")
                return validation_results
            
            # Get model performance
            client = MlflowClient()
            mv = client.get_model_version(model_name, version)
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics
            
            passed_checks = 0
            total_checks = 0
            
            # Validate performance metrics
            for threshold_name, threshold_value in thresholds.items():
                total_checks += 1
                metric_name = threshold_name.replace("min_", "").replace("max_", "")
                
                if metric_name in metrics:
                    actual_value = metrics[metric_name]
                    
                    if threshold_name.startswith("min_"):
                        passed = actual_value >= threshold_value
                        comparison = f">= {threshold_value}"
                    else:  # max_
                        passed = actual_value <= threshold_value
                        comparison = f"<= {threshold_value}"
                    
                    validation_results["checks"][threshold_name] = {
                        "actual": actual_value,
                        "threshold": threshold_value,
                        "passed": passed,
                        "comparison": comparison
                    }
                    
                    if passed:
                        passed_checks += 1
                    else:
                        validation_results["errors"].append(
                            f"{metric_name} ({actual_value:.3f}) does not meet {threshold_name} ({threshold_value})"
                        )
                else:
                    validation_results["warnings"].append(
                        f"Metric {metric_name} not found in model run"
                    )
            
            # Calculate validation score
            if total_checks > 0:
                validation_results["score"] = passed_checks / total_checks
                validation_results["passed"] = validation_results["score"] >= 0.8  # 80% pass rate
            else:
                validation_results["passed"] = True
                validation_results["warnings"].append("No validation checks performed")
            
            logger.info(f"Validation completed: {validation_results['score']:.2%} pass rate")
            return validation_results
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Validation error: {e}")
            return validation_results


class DeploymentManager:
    """Manages model deployments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deployments = {}  # In-memory tracking (would use database in production)
    
    def deploy_model(self, model_name: str, version: str, stage: str) -> bool:
        """Deploy model to specified stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Deployment stage
            
        Returns:
            Success status
        """
        logger.info(f"Deploying model {model_name} v{version} to {stage}")
        
        try:
            deployment_config = self.config["deployment_config"].get(stage.lower(), {})
            
            # Simulate deployment process
            deployment_id = f"{model_name}-{version}-{stage.lower()}"
            
            self.deployments[deployment_id] = {
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "status": "DEPLOYING",
                "deployment_time": datetime.now(timezone.utc).isoformat(),
                "config": deployment_config
            }
            
            # Simulate deployment steps
            time.sleep(1)  # Simulate deployment time
            
            # Mark as deployed
            self.deployments[deployment_id]["status"] = "DEPLOYED"
            
            logger.info(f"Deployment completed: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def get_deployment_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get deployment information."""
        for deployment_id, info in self.deployments.items():
            if info["model_name"] == model_name and info["version"] == version:
                return info
        return None
    
    def check_health(self, model_name: str, version: str) -> Dict[str, Any]:
        """Check deployment health."""
        deployment_info = self.get_deployment_info(model_name, version)
        
        if not deployment_info:
            return {"status": "NOT_DEPLOYED", "healthy": False}
        
        # Simulate health check
        return {
            "status": deployment_info["status"],
            "healthy": deployment_info["status"] == "DEPLOYED",
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    def cleanup_deployment(self, model_name: str, version: str) -> bool:
        """Clean up deployment."""
        deployment_id = f"{model_name}-{version}"
        
        for dep_id in list(self.deployments.keys()):
            if dep_id.startswith(deployment_id):
                del self.deployments[dep_id]
                logger.info(f"Cleaned up deployment: {dep_id}")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model Staging and Lifecycle Management")
    parser.add_argument("--action", required=True,
                       choices=["list", "transition", "promote", "archive", "rollback", "status", "auto-archive"],
                       help="Action to perform")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--stage", help="Target stage")
    parser.add_argument("--from-stage", help="Source stage for promotion")
    parser.add_argument("--to-stage", help="Target stage for promotion")
    parser.add_argument("--description", help="Transition description")
    parser.add_argument("--force", action="store_true", help="Force transition without validation")
    parser.add_argument("--config", default="configs/model_staging_config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize staging manager
    manager = ModelStagingManager(args.config)
    
    try:
        if args.action == "list":
            models = manager.list_models(args.stage)
            
            print(f"Found {len(models)} models")
            for model in models:
                print(f"  {model.name} v{model.version} ({model.stage}) - {model.deployment_status}")
                if model.performance_metrics:
                    metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in model.performance_metrics.items()])
                    print(f"    Performance: {metrics_str}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(model) for model in models], f, indent=2)
                print(f"Results saved to {args.output}")
        
        elif args.action == "transition":
            if not all([args.model_name, args.version, args.stage]):
                print("Error: model-name, version, and stage are required for transition")
                sys.exit(1)
            
            success = manager.transition_model_stage(
                args.model_name, args.version, args.stage,
                args.description or "", args.force
            )
            print(f"Transition {'succeeded' if success else 'failed'}")
        
        elif args.action == "promote":
            if not all([args.model_name, args.from_stage, args.to_stage]):
                print("Error: model-name, from-stage, and to-stage are required for promotion")
                sys.exit(1)
            
            success = manager.promote_model(
                args.model_name, args.from_stage, args.to_stage
            )
            print(f"Promotion {'succeeded' if success else 'failed'}")
        
        elif args.action == "archive":
            if not all([args.model_name, args.version]):
                print("Error: model-name and version are required for archival")
                sys.exit(1)
            
            success = manager.archive_model(
                args.model_name, args.version, args.description or ""
            )
            print(f"Archival {'succeeded' if success else 'failed'}")
        
        elif args.action == "rollback":
            if not args.model_name:
                print("Error: model-name is required for rollback")
                sys.exit(1)
            
            success = manager.rollback_model(args.model_name, args.stage or "Production")
            print(f"Rollback {'succeeded' if success else 'failed'}")
        
        elif args.action == "status":
            if not all([args.model_name, args.version]):
                print("Error: model-name and version are required for status")
                sys.exit(1)
            
            status = manager.get_deployment_status(args.model_name, args.version)
            print(json.dumps(status, indent=2))
        
        elif args.action == "auto-archive":
            archived_count = manager.auto_archive_old_models()
            print(f"Auto-archived {archived_count} models")
    
    except Exception as e:
        logger.error(f"Model staging operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()