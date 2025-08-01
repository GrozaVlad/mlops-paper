#!/usr/bin/env python3
"""
Model Lineage Integration Module

Integrates lineage tracking with existing MLOps components including
MLflow, retraining pipeline, A/B testing, and model staging.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

from model_lineage_tracking import (
    ModelLineageTracker, 
    ModelNode, 
    DataNode,
    LineageRelationType,
    DataLineageType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LineageIntegration:
    """Integrates lineage tracking with MLOps pipeline components"""
    
    def __init__(self, config_path: str = "configs/lineage_config.yaml"):
        """
        Initialize lineage integration
        
        Args:
            config_path: Path to lineage configuration file
        """
        self.config = self._load_config(config_path)
        self.tracker = ModelLineageTracker(
            db_path=self.config['lineage_tracking']['database']['path'],
            mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 
                                         self.config['lineage_tracking']['mlflow']['tracking_uri'])
        )
        self.mlflow_client = MlflowClient()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def track_training_run(self, 
                          run_id: str,
                          model_name: str,
                          version: str,
                          training_data_info: Dict[str, Any],
                          parent_model_id: Optional[str] = None,
                          relationship_type: LineageRelationType = LineageRelationType.DERIVED_FROM) -> ModelNode:
        """
        Track a model training run in the lineage system
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model
            version: Model version
            training_data_info: Information about training data
            parent_model_id: ID of parent model if exists
            relationship_type: Type of relationship to parent
            
        Returns:
            Created ModelNode
        """
        try:
            # Track model from MLflow
            model_node = self.tracker.track_mlflow_model(run_id, model_name, version)
            
            # Add training data
            if training_data_info:
                data_node = DataNode(
                    data_id=training_data_info.get('data_id', f"data_{run_id[:8]}"),
                    name=training_data_info.get('name', 'Training Dataset'),
                    version=training_data_info.get('version', '1.0'),
                    num_samples=training_data_info.get('num_samples'),
                    features=training_data_info.get('features', []),
                    location=training_data_info.get('location'),
                    checksum=training_data_info.get('checksum')
                )
                self.tracker.add_dataset(data_node)
                
                # Add data relationship
                self.tracker.add_model_data_relationship(
                    model_node.model_id,
                    data_node.data_id,
                    DataLineageType.TRAINED_ON
                )
            
            # Add parent relationship if exists
            if parent_model_id:
                self.tracker.add_model_relationship(
                    parent_model_id,
                    model_node.model_id,
                    relationship_type
                )
            
            logger.info(f"Tracked training run {run_id} in lineage system")
            return model_node
            
        except Exception as e:
            logger.error(f"Error tracking training run: {str(e)}")
            raise
    
    def track_retraining(self,
                        old_model_id: str,
                        new_run_id: str,
                        model_name: str,
                        new_version: str,
                        trigger_reason: str,
                        retraining_data_info: Optional[Dict[str, Any]] = None) -> ModelNode:
        """
        Track a model retraining event
        
        Args:
            old_model_id: ID of the model being retrained
            new_run_id: MLflow run ID of new model
            model_name: Name of the model
            new_version: New model version
            trigger_reason: Reason for retraining
            retraining_data_info: Information about retraining data
            
        Returns:
            Created ModelNode for retrained model
        """
        # Track the new model
        new_model = self.tracker.track_mlflow_model(new_run_id, model_name, new_version)
        
        # Add retraining relationship
        self.tracker.add_model_relationship(
            old_model_id,
            new_model.model_id,
            LineageRelationType.RETRAINED_FROM,
            metadata={
                'trigger_reason': trigger_reason,
                'retrained_at': datetime.utcnow().isoformat()
            }
        )
        
        # Track retraining data if provided
        if retraining_data_info:
            data_node = DataNode(
                data_id=retraining_data_info.get('data_id', f"retrain_data_{new_run_id[:8]}"),
                name=retraining_data_info.get('name', 'Retraining Dataset'),
                version=retraining_data_info.get('version', '1.0'),
                num_samples=retraining_data_info.get('num_samples'),
                location=retraining_data_info.get('location')
            )
            self.tracker.add_dataset(data_node)
            
            self.tracker.add_model_data_relationship(
                new_model.model_id,
                data_node.data_id,
                DataLineageType.TRAINED_ON
            )
        
        logger.info(f"Tracked retraining from {old_model_id} to {new_model.model_id}")
        return new_model
    
    def track_ab_test(self,
                     champion_model_id: str,
                     challenger_model_id: str,
                     experiment_id: str,
                     test_config: Dict[str, Any]) -> None:
        """
        Track A/B test relationship between models
        
        Args:
            champion_model_id: ID of champion model
            challenger_model_id: ID of challenger model
            experiment_id: A/B test experiment ID
            test_config: A/B test configuration
        """
        # Add A/B test relationship
        self.tracker.add_model_relationship(
            champion_model_id,
            challenger_model_id,
            LineageRelationType.A_B_TEST_VARIANT,
            metadata={
                'experiment_id': experiment_id,
                'traffic_split': test_config.get('traffic_percentage', 50),
                'success_criteria': test_config.get('success_criteria', {}),
                'started_at': datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Tracked A/B test between {champion_model_id} and {challenger_model_id}")
    
    def track_model_promotion(self,
                            model_id: str,
                            from_stage: str,
                            to_stage: str,
                            promoted_by: str,
                            reason: Optional[str] = None) -> None:
        """
        Track model stage promotion in lineage
        
        Args:
            model_id: ID of model being promoted
            from_stage: Previous stage
            to_stage: New stage
            promoted_by: User who promoted the model
            reason: Reason for promotion
        """
        if model_id in self.tracker.model_graph:
            model = self.tracker.model_graph.nodes[model_id]['data']
            model.stage = to_stage
            model.tags['last_stage_transition'] = f"{from_stage}->{to_stage}"
            model.tags['promoted_by'] = promoted_by
            model.tags['promotion_time'] = datetime.utcnow().isoformat()
            if reason:
                model.tags['promotion_reason'] = reason
            
            # Re-add to update in database
            self.tracker.add_model(model)
            
            logger.info(f"Tracked promotion of {model_id} from {from_stage} to {to_stage}")
    
    def track_ensemble_models(self,
                            ensemble_model_id: str,
                            member_model_ids: List[str],
                            ensemble_config: Dict[str, Any]) -> None:
        """
        Track ensemble model relationships
        
        Args:
            ensemble_model_id: ID of the ensemble model
            member_model_ids: List of member model IDs
            ensemble_config: Ensemble configuration
        """
        for member_id in member_model_ids:
            self.tracker.add_model_relationship(
                member_id,
                ensemble_model_id,
                LineageRelationType.ENSEMBLE_MEMBER,
                metadata={
                    'weight': ensemble_config.get('weights', {}).get(member_id, 1.0),
                    'ensemble_type': ensemble_config.get('type', 'voting')
                }
            )
        
        logger.info(f"Tracked ensemble {ensemble_model_id} with {len(member_model_ids)} members")
    
    def track_data_drift_relationship(self,
                                    model_id: str,
                                    drift_detection_data: Dict[str, Any]) -> None:
        """
        Track data drift detection results in lineage
        
        Args:
            model_id: ID of model experiencing drift
            drift_detection_data: Drift detection results
        """
        # Create a data node for drift detection
        drift_data = DataNode(
            data_id=f"drift_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name="Drift Detection Dataset",
            version="1.0",
            statistics={
                'drift_score': drift_detection_data.get('drift_score'),
                'drift_features': drift_detection_data.get('drifted_features', []),
                'detection_method': drift_detection_data.get('method', 'statistical')
            }
        )
        self.tracker.add_dataset(drift_data)
        
        # Add relationship
        self.tracker.add_model_data_relationship(
            model_id,
            drift_data.data_id,
            DataLineageType.VALIDATED_ON,
            metadata={'drift_detected': True}
        )
        
        logger.info(f"Tracked data drift for model {model_id}")
    
    def get_model_impact_analysis(self, model_id: str) -> Dict[str, Any]:
        """
        Analyze the impact of a model on downstream models
        
        Args:
            model_id: ID of model to analyze
            
        Returns:
            Impact analysis results
        """
        descendants = self.tracker.get_model_descendants(model_id)
        
        impact_analysis = {
            'model_id': model_id,
            'total_descendants': len(descendants),
            'immediate_impact': [],
            'downstream_impact': [],
            'production_impact': [],
            'metrics_comparison': {}
        }
        
        # Analyze immediate children
        for child_id in self.tracker.model_graph.successors(model_id):
            child = self.tracker.model_graph.nodes[child_id]['data']
            impact_analysis['immediate_impact'].append({
                'model_id': child_id,
                'model_name': child.model_name,
                'stage': child.stage,
                'metrics': child.metrics
            })
            
            if child.stage == 'Production':
                impact_analysis['production_impact'].append(child_id)
        
        # Analyze all descendants
        for desc_id in descendants:
            desc = self.tracker.model_graph.nodes[desc_id]['data']
            impact_analysis['downstream_impact'].append({
                'model_id': desc_id,
                'model_name': desc.model_name,
                'stage': desc.stage,
                'generation': self.tracker._get_model_generation(desc_id)
            })
        
        # Compare metrics if available
        if model_id in self.tracker.model_graph:
            base_metrics = self.tracker.model_graph.nodes[model_id]['data'].metrics
            for metric_name in base_metrics:
                impact_analysis['metrics_comparison'][metric_name] = {
                    'base_value': base_metrics[metric_name],
                    'descendant_values': []
                }
                
                for desc_id in descendants:
                    desc_metrics = self.tracker.model_graph.nodes[desc_id]['data'].metrics
                    if metric_name in desc_metrics:
                        impact_analysis['metrics_comparison'][metric_name]['descendant_values'].append({
                            'model_id': desc_id,
                            'value': desc_metrics[metric_name],
                            'improvement': desc_metrics[metric_name] - base_metrics[metric_name]
                        })
        
        return impact_analysis
    
    def generate_compliance_report(self, model_id: str) -> Dict[str, Any]:
        """
        Generate a compliance report for model lineage
        
        Args:
            model_id: ID of model to report on
            
        Returns:
            Compliance report
        """
        lineage_report = self.tracker.generate_lineage_report(model_id)
        
        compliance_report = {
            'model_id': model_id,
            'report_date': datetime.utcnow().isoformat(),
            'model_info': lineage_report['model_info'],
            'data_provenance': lineage_report['data_lineage'],
            'model_provenance': {
                'ancestors': lineage_report['ancestry']['all_ancestors'],
                'training_lineage': [],
                'validation_history': []
            },
            'audit_trail': [],
            'compliance_checks': {
                'data_privacy': True,
                'model_bias': 'not_evaluated',
                'reproducibility': True,
                'documentation': True
            }
        }
        
        # Build audit trail
        for rel_type, relationships in lineage_report['relationships'].items():
            for rel in relationships:
                compliance_report['audit_trail'].append({
                    'timestamp': rel['created_at'],
                    'action': rel_type,
                    'source': rel['source'],
                    'target': rel['target'],
                    'metadata': rel.get('metadata', {})
                })
        
        # Sort audit trail by timestamp
        compliance_report['audit_trail'].sort(key=lambda x: x['timestamp'])
        
        return compliance_report
    
    def cleanup_old_lineage(self, retention_days: int = 365) -> Dict[str, int]:
        """
        Clean up old lineage data based on retention policy
        
        Args:
            retention_days: Number of days to retain lineage data
            
        Returns:
            Cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        stats = {
            'models_removed': 0,
            'datasets_removed': 0,
            'relationships_removed': 0
        }
        
        # Find old archived models
        models_to_remove = []
        for node_id in self.tracker.model_graph.nodes():
            model = self.tracker.model_graph.nodes[node_id]['data']
            if model.stage == 'Archived' and model.created_at < cutoff_date:
                # Check if it has no production descendants
                descendants = self.tracker.get_model_descendants(node_id)
                has_prod_descendant = any(
                    self.tracker.model_graph.nodes[d]['data'].stage == 'Production'
                    for d in descendants if d in self.tracker.model_graph
                )
                
                if not has_prod_descendant:
                    models_to_remove.append(node_id)
        
        # Remove old models
        for model_id in models_to_remove:
            self.tracker.model_graph.remove_node(model_id)
            stats['models_removed'] += 1
        
        logger.info(f"Cleaned up {stats['models_removed']} old models")
        return stats


def setup_lineage_hooks():
    """Set up automatic lineage tracking hooks"""
    
    integration = LineageIntegration()
    
    # Hook into MLflow model registry
    def on_model_registered(registered_model):
        """Track model registration in lineage"""
        try:
            latest_version = registered_model.latest_versions[0]
            run_id = latest_version.run_id
            
            integration.track_training_run(
                run_id=run_id,
                model_name=registered_model.name,
                version=latest_version.version,
                training_data_info={
                    'name': 'Registered Model Training Data',
                    'version': '1.0'
                }
            )
        except Exception as e:
            logger.error(f"Error in model registration hook: {str(e)}")
    
    # Hook into stage transitions
    def on_stage_transition(model_name, version, stage, archive_existing_versions):
        """Track stage transitions in lineage"""
        try:
            model_id = f"{model_name}_v{version}"
            integration.track_model_promotion(
                model_id=model_id,
                from_stage="None",  # Would need to track previous state
                to_stage=stage,
                promoted_by="system"
            )
        except Exception as e:
            logger.error(f"Error in stage transition hook: {str(e)}")
    
    return integration


def main():
    """Example usage of lineage integration"""
    
    # Initialize integration
    integration = LineageIntegration()
    
    # Example: Track a training run
    model = integration.track_training_run(
        run_id="test_run_123",
        model_name="DrugBAN",
        version="2.0",
        training_data_info={
            'data_id': 'biosnap_2023',
            'name': 'BIOSNAP 2023 Dataset',
            'version': '2023.1',
            'num_samples': 150000,
            'features': ['smiles', 'target_sequence'],
            'location': 's3://mlops-data/biosnap-2023/'
        }
    )
    
    print(f"Tracked model: {model.model_id}")
    
    # Example: Track retraining
    retrained_model = integration.track_retraining(
        old_model_id=model.model_id,
        new_run_id="retrain_run_456",
        model_name="DrugBAN",
        new_version="2.1",
        trigger_reason="performance_degradation"
    )
    
    print(f"Tracked retrained model: {retrained_model.model_id}")
    
    # Generate impact analysis
    impact = integration.get_model_impact_analysis(model.model_id)
    print(f"Impact analysis: {impact}")
    
    # Generate compliance report
    compliance = integration.generate_compliance_report(retrained_model.model_id)
    print(f"Compliance report generated with {len(compliance['audit_trail'])} audit entries")


if __name__ == "__main__":
    main()