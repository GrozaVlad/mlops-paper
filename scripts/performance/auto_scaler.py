#!/usr/bin/env python3
"""
Auto Scaler

Implements automatic scaling decisions based on performance metrics and load patterns.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import kubernetes
from kubernetes import client, config
import boto3
from botocore.exceptions import ClientError
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    OUT = "out"
    IN = "in"
    NONE = "none"


class ScalingStrategy(Enum):
    """Scaling strategies"""
    REACTIVE = "reactive"          # React to current metrics
    PREDICTIVE = "predictive"      # Scale based on predicted load
    SCHEDULED = "scheduled"        # Pre-scheduled scaling
    HYBRID = "hybrid"              # Combination of strategies


@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    decision_id: str
    resource_type: str
    resource_name: str
    current_replicas: int
    target_replicas: int
    direction: ScalingDirection
    strategy: ScalingStrategy
    reason: str
    confidence: float
    metrics_snapshot: Dict[str, float]
    timestamp: datetime
    executed: bool = False
    execution_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class LoadPrediction:
    """Load prediction for capacity planning"""
    timestamp: datetime
    predicted_cpu: float
    predicted_memory: float
    predicted_requests_per_second: float
    confidence_interval: Tuple[float, float]
    prediction_horizon_minutes: int


class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self, 
                 config_path: str = "configs/performance_config.yaml",
                 performance_monitor=None):
        """
        Initialize auto scaler
        
        Args:
            config_path: Path to configuration file
            performance_monitor: Performance monitor instance
        """
        self.config = self._load_config(config_path)
        self.performance_monitor = performance_monitor
        
        # Scaling state
        self.scaling_decisions = []
        self.last_scaling_time = {}
        self.load_predictions = deque(maxlen=100)
        
        # Initialize clients
        self._init_clients()
        
        # Scaling parameters
        self.scaling_config = self.config['performance_monitoring']['scaling']
        self.cooldown_minutes = self.scaling_config['cooldown_minutes']
        
        # Historical data for prediction
        self.historical_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Auto-scaling state
        self.auto_scaling_active = False
        self.scaling_thread = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_clients(self):
        """Initialize Kubernetes and cloud clients"""
        try:
            # Kubernetes client
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
                
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_autoscaling_v1 = client.AutoscalingV1Api()
            self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
            
        except Exception as e:
            logger.warning(f"Kubernetes client initialization failed: {str(e)}")
            self.k8s_apps_v1 = None
            
        try:
            # AWS Auto Scaling client (for EKS node groups)
            self.autoscaling_client = boto3.client('autoscaling')
            self.eks_client = boto3.client('eks')
            
        except Exception as e:
            logger.warning(f"AWS client initialization failed: {str(e)}")
            self.autoscaling_client = None
    
    def start_auto_scaling(self):
        """Start automatic scaling"""
        if self.auto_scaling_active:
            logger.warning("Auto scaling already active")
            return
        
        self.auto_scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        logger.info("Auto scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling"""
        self.auto_scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join()
        
        logger.info("Auto scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop"""
        while self.auto_scaling_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                
                # Store historical data
                self._store_historical_metrics(current_metrics)
                
                # Generate load predictions
                predictions = self._generate_load_predictions()
                
                # Make scaling decisions
                decisions = self._make_scaling_decisions(current_metrics, predictions)
                
                # Execute scaling decisions
                for decision in decisions:
                    if self._should_execute_decision(decision):
                        self._execute_scaling_decision(decision)
                
                # Cleanup old decisions
                self._cleanup_old_decisions()
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {str(e)}")
            
            # Wait before next iteration
            time.sleep(self.config['performance_monitoring']['monitoring']['interval_seconds'])
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        metrics = {}
        
        try:
            # Get metrics from performance monitor if available
            if self.performance_monitor:
                summary = self.performance_monitor.get_performance_summary()
                for metric_name, metric_data in summary['current_metrics'].items():
                    metrics[metric_name] = metric_data['value']
            
            # Get Kubernetes metrics
            k8s_metrics = self._get_kubernetes_metrics()
            metrics.update(k8s_metrics)
            
            # Get application metrics
            app_metrics = self._get_application_metrics()
            metrics.update(app_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting current metrics: {str(e)}")
        
        return metrics
    
    def _get_kubernetes_metrics(self) -> Dict[str, float]:
        """Get Kubernetes-specific metrics"""
        metrics = {}
        
        if not self.k8s_apps_v1:
            return metrics
        
        try:
            # Get deployment metrics
            deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace="drugban")
            
            for deployment in deployments.items:
                name = deployment.metadata.name
                replicas = deployment.spec.replicas or 0
                ready_replicas = deployment.status.ready_replicas or 0
                
                metrics[f"k8s_{name}_replicas"] = replicas
                metrics[f"k8s_{name}_ready_replicas"] = ready_replicas
                metrics[f"k8s_{name}_replica_ratio"] = ready_replicas / max(replicas, 1)
            
            # Get node metrics (if available)
            nodes = self.k8s_core_v1.list_node()
            total_nodes = len(nodes.items)
            ready_nodes = sum(1 for node in nodes.items 
                            for condition in node.status.conditions
                            if condition.type == "Ready" and condition.status == "True")
            
            metrics["k8s_total_nodes"] = total_nodes
            metrics["k8s_ready_nodes"] = ready_nodes
            metrics["k8s_node_ready_ratio"] = ready_nodes / max(total_nodes, 1)
            
        except Exception as e:
            logger.error(f"Error getting Kubernetes metrics: {str(e)}")
        
        return metrics
    
    def _get_application_metrics(self) -> Dict[str, float]:
        """Get application-specific metrics"""
        metrics = {}
        
        try:
            # API server metrics
            api_config = self.config['performance_monitoring']['components']['api_server']
            if api_config['enabled']:
                metrics_url = api_config['metrics_endpoint']
                
                try:
                    response = requests.get(metrics_url, timeout=5)
                    if response.status_code == 200:
                        # Parse Prometheus metrics
                        app_metrics = self._parse_prometheus_metrics(response.text)
                        metrics.update(app_metrics)
                except requests.RequestException:
                    pass  # Metrics endpoint not available
            
        except Exception as e:
            logger.error(f"Error getting application metrics: {str(e)}")
        
        return metrics
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format"""
        metrics = {}
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#'):
                try:
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[f"app_{metric_name}"] = metric_value
                except:
                    continue
        return metrics
    
    def _store_historical_metrics(self, metrics: Dict[str, float]):
        """Store metrics for historical analysis"""
        timestamp = datetime.utcnow()
        
        for metric_name, value in metrics.items():
            self.historical_metrics[metric_name].append({
                'timestamp': timestamp,
                'value': value
            })
    
    def _generate_load_predictions(self) -> List[LoadPrediction]:
        """Generate load predictions for proactive scaling"""
        predictions = []
        
        try:
            # Simple time-series prediction based on historical data
            for horizon in [5, 15, 30]:  # 5, 15, 30 minute predictions
                prediction = self._predict_load(horizon)
                if prediction:
                    predictions.append(prediction)
                    self.load_predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Error generating load predictions: {str(e)}")
        
        return predictions
    
    def _predict_load(self, horizon_minutes: int) -> Optional[LoadPrediction]:
        """Predict load for a specific time horizon"""
        try:
            # Get historical CPU data
            cpu_data = self.historical_metrics.get('cpu_utilization', [])
            if len(cpu_data) < 10:  # Need enough data points
                return None
            
            # Simple moving average prediction (can be enhanced with ML models)
            recent_values = [d['value'] for d in list(cpu_data)[-10:]]
            predicted_cpu = np.mean(recent_values)
            
            # Add some trend analysis
            if len(recent_values) >= 5:
                early_avg = np.mean(recent_values[:5])
                late_avg = np.mean(recent_values[-5:])
                trend = (late_avg - early_avg) / 5  # Trend per data point
                
                # Project trend forward
                predicted_cpu += trend * (horizon_minutes / 5)  # Assuming 5-minute intervals
            
            # Similar for memory (simplified)
            memory_data = self.historical_metrics.get('memory_utilization', [])
            predicted_memory = predicted_cpu * 0.8  # Rough correlation
            
            # Request rate prediction (simplified)
            request_data = self.historical_metrics.get('app_requests_per_second', [])
            if request_data:
                recent_requests = [d['value'] for d in list(request_data)[-5:]]
                predicted_requests = np.mean(recent_requests) if recent_requests else 50
            else:
                predicted_requests = 50  # Default
            
            # Confidence interval (simplified)
            std_dev = np.std(recent_values) if len(recent_values) > 1 else 10
            confidence_interval = (
                max(0, predicted_cpu - 1.96 * std_dev),  # 95% CI lower bound
                min(100, predicted_cpu + 1.96 * std_dev)  # 95% CI upper bound
            )
            
            return LoadPrediction(
                timestamp=datetime.utcnow() + timedelta(minutes=horizon_minutes),
                predicted_cpu=predicted_cpu,
                predicted_memory=predicted_memory,
                predicted_requests_per_second=predicted_requests,
                confidence_interval=confidence_interval,
                prediction_horizon_minutes=horizon_minutes
            )
            
        except Exception as e:
            logger.error(f"Error predicting load: {str(e)}")
            return None
    
    def _make_scaling_decisions(self, 
                               current_metrics: Dict[str, float],
                               predictions: List[LoadPrediction]) -> List[ScalingDecision]:
        """Make scaling decisions based on metrics and predictions"""
        decisions = []
        
        try:
            # Get scaling policies
            policies = self.scaling_config.get('policies', {})
            
            # CPU-based scaling
            if policies.get('cpu_based', {}).get('enabled', True):
                cpu_decision = self._make_cpu_scaling_decision(current_metrics, predictions)
                if cpu_decision:
                    decisions.append(cpu_decision)
            
            # Memory-based scaling
            if policies.get('memory_based', {}).get('enabled', True):
                memory_decision = self._make_memory_scaling_decision(current_metrics, predictions)
                if memory_decision:
                    decisions.append(memory_decision)
            
            # Request rate-based scaling
            if policies.get('request_rate_based', {}).get('enabled', True):
                request_decision = self._make_request_rate_scaling_decision(current_metrics, predictions)
                if request_decision:
                    decisions.append(request_decision)
            
            # Queue length-based scaling
            if policies.get('queue_length_based', {}).get('enabled', True):
                queue_decision = self._make_queue_length_scaling_decision(current_metrics)
                if queue_decision:
                    decisions.append(queue_decision)
            
        except Exception as e:
            logger.error(f"Error making scaling decisions: {str(e)}")
        
        return decisions
    
    def _make_cpu_scaling_decision(self, 
                                  metrics: Dict[str, float],
                                  predictions: List[LoadPrediction]) -> Optional[ScalingDecision]:
        """Make CPU-based scaling decision"""
        try:
            cpu_policy = self.scaling_config['policies']['cpu_based']
            current_cpu = metrics.get('cpu_utilization', 0)
            
            # Get current replicas for main deployment
            current_replicas = self._get_current_replicas("drugban-api")
            
            # Reactive scaling based on current CPU
            if current_cpu > cpu_policy['scale_up_threshold']:
                target_replicas = min(current_replicas + 1, self.scaling_config['max_instances'])
                direction = ScalingDirection.OUT
                strategy = ScalingStrategy.REACTIVE
                reason = f"High CPU utilization: {current_cpu:.1f}%"
                confidence = 0.8
                
            elif current_cpu < cpu_policy['scale_down_threshold']:
                target_replicas = max(current_replicas - 1, self.scaling_config['min_instances'])
                direction = ScalingDirection.IN
                strategy = ScalingStrategy.REACTIVE
                reason = f"Low CPU utilization: {current_cpu:.1f}%"
                confidence = 0.7
                
            else:
                # Check predictions for proactive scaling
                future_cpu = None
                for prediction in predictions:
                    if prediction.prediction_horizon_minutes == 15:  # 15-minute lookahead
                        future_cpu = prediction.predicted_cpu
                        break
                
                if future_cpu and future_cpu > cpu_policy['scale_up_threshold']:
                    target_replicas = min(current_replicas + 1, self.scaling_config['max_instances'])
                    direction = ScalingDirection.OUT
                    strategy = ScalingStrategy.PREDICTIVE
                    reason = f"Predicted high CPU: {future_cpu:.1f}% in 15 minutes"
                    confidence = 0.6
                else:
                    return None
            
            if target_replicas == current_replicas:
                return None
            
            return ScalingDecision(
                decision_id=f"cpu_scale_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                resource_type="deployment",
                resource_name="drugban-api",
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                direction=direction,
                strategy=strategy,
                reason=reason,
                confidence=confidence,
                metrics_snapshot=metrics.copy(),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error making CPU scaling decision: {str(e)}")
            return None
    
    def _make_memory_scaling_decision(self,
                                     metrics: Dict[str, float],
                                     predictions: List[LoadPrediction]) -> Optional[ScalingDecision]:
        """Make memory-based scaling decision"""
        try:
            memory_policy = self.scaling_config['policies']['memory_based']
            current_memory = metrics.get('memory_utilization', 0)
            
            current_replicas = self._get_current_replicas("drugban-api")
            
            if current_memory > memory_policy['scale_up_threshold']:
                target_replicas = min(current_replicas + 1, self.scaling_config['max_instances'])
                
                return ScalingDecision(
                    decision_id=f"memory_scale_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    resource_type="deployment",
                    resource_name="drugban-api",
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    direction=ScalingDirection.OUT,
                    strategy=ScalingStrategy.REACTIVE,
                    reason=f"High memory utilization: {current_memory:.1f}%",
                    confidence=0.8,
                    metrics_snapshot=metrics.copy(),
                    timestamp=datetime.utcnow()
                )
            
        except Exception as e:
            logger.error(f"Error making memory scaling decision: {str(e)}")
        
        return None
    
    def _make_request_rate_scaling_decision(self,
                                           metrics: Dict[str, float],
                                           predictions: List[LoadPrediction]) -> Optional[ScalingDecision]:
        """Make request rate-based scaling decision"""
        try:
            request_policy = self.scaling_config['policies']['request_rate_based']
            current_rps = metrics.get('app_requests_per_second_total', 0)
            
            current_replicas = self._get_current_replicas("drugban-api")
            requests_per_replica = current_rps / max(current_replicas, 1)
            
            if requests_per_replica > request_policy['scale_up_threshold']:
                target_replicas = min(current_replicas + 1, self.scaling_config['max_instances'])
                
                return ScalingDecision(
                    decision_id=f"request_scale_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    resource_type="deployment",
                    resource_name="drugban-api",
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    direction=ScalingDirection.OUT,
                    strategy=ScalingStrategy.REACTIVE,
                    reason=f"High request rate: {requests_per_replica:.1f} RPS per replica",
                    confidence=0.9,
                    metrics_snapshot=metrics.copy(),
                    timestamp=datetime.utcnow()
                )
            
            elif requests_per_replica < request_policy['scale_down_threshold']:
                target_replicas = max(current_replicas - 1, self.scaling_config['min_instances'])
                
                if target_replicas < current_replicas:
                    return ScalingDecision(
                        decision_id=f"request_scale_down_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_type="deployment",
                        resource_name="drugban-api",
                        current_replicas=current_replicas,
                        target_replicas=target_replicas,
                        direction=ScalingDirection.IN,
                        strategy=ScalingStrategy.REACTIVE,
                        reason=f"Low request rate: {requests_per_replica:.1f} RPS per replica",
                        confidence=0.7,
                        metrics_snapshot=metrics.copy(),
                        timestamp=datetime.utcnow()
                    )
            
        except Exception as e:
            logger.error(f"Error making request rate scaling decision: {str(e)}")
        
        return None
    
    def _make_queue_length_scaling_decision(self, metrics: Dict[str, float]) -> Optional[ScalingDecision]:
        """Make queue length-based scaling decision"""
        try:
            queue_policy = self.scaling_config['policies']['queue_length_based']
            queue_length = metrics.get('queue_length', 0)
            
            current_replicas = self._get_current_replicas("drugban-api")
            
            if queue_length > queue_policy['scale_up_threshold']:
                target_replicas = min(current_replicas + 2, self.scaling_config['max_instances'])  # Scale faster for queues
                
                return ScalingDecision(
                    decision_id=f"queue_scale_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    resource_type="deployment",
                    resource_name="drugban-api",
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    direction=ScalingDirection.OUT,
                    strategy=ScalingStrategy.REACTIVE,
                    reason=f"High queue length: {queue_length}",
                    confidence=0.9,
                    metrics_snapshot=metrics.copy(),
                    timestamp=datetime.utcnow()
                )
            
        except Exception as e:
            logger.error(f"Error making queue length scaling decision: {str(e)}")
        
        return None
    
    def _get_current_replicas(self, deployment_name: str) -> int:
        """Get current number of replicas for a deployment"""
        if not self.k8s_apps_v1:
            return 1
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace="drugban"
            )
            return deployment.spec.replicas or 1
        except Exception as e:
            logger.error(f"Error getting current replicas: {str(e)}")
            return 1
    
    def _should_execute_decision(self, decision: ScalingDecision) -> bool:
        """Check if a scaling decision should be executed"""
        try:
            # Check cooldown period
            last_scaling = self.last_scaling_time.get(decision.resource_name)
            if last_scaling:
                time_since_last = datetime.utcnow() - last_scaling
                if time_since_last < timedelta(minutes=self.cooldown_minutes):
                    logger.info(f"Scaling decision {decision.decision_id} blocked by cooldown")
                    return False
            
            # Check confidence threshold
            min_confidence = 0.5  # Minimum confidence to execute
            if decision.confidence < min_confidence:
                logger.info(f"Scaling decision {decision.decision_id} blocked by low confidence: {decision.confidence}")
                return False
            
            # Check if there's already a pending decision for the same resource
            pending_decisions = [
                d for d in self.scaling_decisions
                if d.resource_name == decision.resource_name and not d.executed
            ]
            if pending_decisions:
                logger.info(f"Scaling decision {decision.decision_id} blocked by pending decision")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if decision should be executed: {str(e)}")
            return False
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision"""
        try:
            logger.info(f"Executing scaling decision: {decision.reason}")
            
            if decision.resource_type == "deployment":
                success = self._scale_kubernetes_deployment(
                    decision.resource_name,
                    decision.target_replicas
                )
            else:
                logger.warning(f"Unknown resource type: {decision.resource_type}")
                success = False
            
            # Update decision status
            decision.executed = True
            decision.execution_time = datetime.utcnow()
            decision.success = success
            
            if success:
                self.last_scaling_time[decision.resource_name] = datetime.utcnow()
                logger.info(f"Successfully executed scaling decision {decision.decision_id}")
            else:
                logger.error(f"Failed to execute scaling decision {decision.decision_id}")
            
            # Add to history
            self.scaling_decisions.append(decision)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing scaling decision: {str(e)}")
            decision.executed = True
            decision.execution_time = datetime.utcnow()
            decision.success = False
            decision.error_message = str(e)
            self.scaling_decisions.append(decision)
            return False
    
    def _scale_kubernetes_deployment(self, deployment_name: str, target_replicas: int) -> bool:
        """Scale a Kubernetes deployment"""
        if not self.k8s_apps_v1:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Scale the deployment
            self.k8s_apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace="drugban",
                body={"spec": {"replicas": target_replicas}}
            )
            
            logger.info(f"Scaled {deployment_name} to {target_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling Kubernetes deployment: {str(e)}")
            return False
    
    def _cleanup_old_decisions(self):
        """Clean up old scaling decisions"""
        try:
            # Keep decisions for 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            self.scaling_decisions = [
                decision for decision in self.scaling_decisions
                if decision.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old decisions: {str(e)}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        try:
            recent_decisions = [
                d for d in self.scaling_decisions
                if d.timestamp > datetime.utcnow() - timedelta(hours=1)
            ]
            
            successful_decisions = [d for d in recent_decisions if d.success]
            failed_decisions = [d for d in recent_decisions if d.executed and not d.success]
            
            status = {
                'auto_scaling_active': self.auto_scaling_active,
                'recent_decisions_count': len(recent_decisions),
                'successful_decisions_count': len(successful_decisions),
                'failed_decisions_count': len(failed_decisions),
                'last_scaling_times': {
                    resource: time.isoformat()
                    for resource, time in self.last_scaling_time.items()
                },
                'current_replicas': {},
                'recent_decisions': [
                    {
                        'decision_id': d.decision_id,
                        'resource_name': d.resource_name,
                        'direction': d.direction.value,
                        'strategy': d.strategy.value,
                        'reason': d.reason,
                        'confidence': d.confidence,
                        'executed': d.executed,
                        'success': d.success,
                        'timestamp': d.timestamp.isoformat()
                    }
                    for d in recent_decisions
                ]
            }
            
            # Get current replicas for monitored deployments
            try:
                deployments = ["drugban-api"]  # Add more as needed
                for deployment in deployments:
                    status['current_replicas'][deployment] = self._get_current_replicas(deployment)
            except:
                pass
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting scaling status: {str(e)}")
            return {
                'auto_scaling_active': self.auto_scaling_active,
                'error': str(e)
            }
    
    def create_hpa(self, deployment_name: str, min_replicas: int = 2, max_replicas: int = 10) -> bool:
        """Create Horizontal Pod Autoscaler"""
        if not self.k8s_autoscaling_v2:
            logger.error("Kubernetes autoscaling API not available")
            return False
        
        try:
            hpa_spec = client.V2HorizontalPodAutoscaler(
                api_version="autoscaling/v2",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa",
                    namespace="drugban"
                ),
                spec=client.V2HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V2CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    metrics=[
                        client.V2MetricSpec(
                            type="Resource",
                            resource=client.V2ResourceMetricSource(
                                name="cpu",
                                target=client.V2MetricTarget(
                                    type="Utilization",
                                    average_utilization=70
                                )
                            )
                        ),
                        client.V2MetricSpec(
                            type="Resource",
                            resource=client.V2ResourceMetricSource(
                                name="memory",
                                target=client.V2MetricTarget(
                                    type="Utilization",
                                    average_utilization=80
                                )
                            )
                        )
                    ]
                )
            )
            
            self.k8s_autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace="drugban",
                body=hpa_spec
            )
            
            logger.info(f"Created HPA for {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating HPA: {str(e)}")
            return False
    
    def optimize_resource_requests(self, deployment_name: str) -> bool:
        """Optimize resource requests based on historical usage"""
        try:
            # Analyze historical CPU and memory usage
            cpu_data = [d['value'] for d in self.historical_metrics.get('cpu_utilization', [])]
            memory_data = [d['value'] for d in self.historical_metrics.get('memory_utilization', [])]
            
            if not cpu_data or not memory_data:
                logger.warning("Insufficient historical data for optimization")
                return False
            
            # Calculate optimal resource requests (P95 of usage)
            cpu_p95 = np.percentile(cpu_data, 95)
            memory_p95 = np.percentile(memory_data, 95)
            
            # Convert to resource requests (add some buffer)
            cpu_request = int(cpu_p95 * 1.2 * 10)  # millicores
            memory_request = int(memory_p95 * 1.2 * 1024)  # MiB
            
            logger.info(f"Recommended resources for {deployment_name}: CPU {cpu_request}m, Memory {memory_request}Mi")
            
            # Note: Actual patching of deployment would require more complex logic
            # This is a simplified version showing the concept
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing resource requests: {str(e)}")
            return False


def main():
    """Example usage of auto scaler"""
    
    # Initialize auto scaler
    scaler = AutoScaler()
    
    # Start auto scaling
    scaler.start_auto_scaling()
    
    try:
        # Let it run for a bit
        time.sleep(120)  # 2 minutes
        
        # Get status
        status = scaler.get_scaling_status()
        print("\nAuto Scaling Status:")
        print(f"Active: {status['auto_scaling_active']}")
        print(f"Recent decisions: {status['recent_decisions_count']}")
        print(f"Successful: {status['successful_decisions_count']}")
        print(f"Failed: {status['failed_decisions_count']}")
        
        # Show recent decisions
        for decision in status['recent_decisions']:
            print(f"- {decision['reason']} ({decision['strategy']}) - {'Success' if decision['success'] else 'Failed'}")
        
        # Create HPA for main deployment
        scaler.create_hpa("drugban-api", min_replicas=2, max_replicas=10)
        
    finally:
        # Stop auto scaling
        scaler.stop_auto_scaling()


if __name__ == "__main__":
    main()