#!/usr/bin/env python3
"""
Performance Monitor and Optimizer

Monitors system performance and automatically adjusts scaling and configuration
for optimal performance in the MLOps pipeline.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import boto3
from botocore.exceptions import ClientError
import kubernetes
from kubernetes import client, config
import redis
import requests
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    REQUEST_LATENCY = "request_latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    MODEL_INFERENCE_TIME = "model_inference_time"
    BATCH_PROCESSING_TIME = "batch_processing_time"
    DATABASE_RESPONSE_TIME = "database_response_time"
    CACHE_HIT_RATE = "cache_hit_rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    OPTIMIZE = "optimize"
    RESTART = "restart"


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Represents a performance alert"""
    alert_id: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    threshold: float
    current_value: float
    source: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ScalingRecommendation:
    """Represents a scaling recommendation"""
    recommendation_id: str
    resource_type: str
    resource_id: str
    action: ScalingAction
    reason: str
    current_state: Dict[str, Any]
    recommended_state: Dict[str, Any]
    estimated_impact: Dict[str, float]
    confidence: float
    priority: int


class PerformanceMonitor:
    """Comprehensive performance monitoring and optimization system"""
    
    def __init__(self, config_path: str = "configs/performance_config.yaml"):
        """
        Initialize performance monitor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.recommendations = []
        
        # Initialize clients
        self._init_clients()
        
        # Performance thresholds
        self.thresholds = self.config.get('thresholds', {})
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Historical data
        self.performance_history = defaultdict(list)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'monitoring': {
                'interval_seconds': 30,
                'retention_hours': 24,
                'enable_auto_scaling': True,
                'enable_alerts': True
            },
            'thresholds': {
                'cpu_utilization_high': 80,
                'cpu_utilization_low': 20,
                'memory_utilization_high': 85,
                'memory_utilization_low': 30,
                'request_latency_high': 2000,  # milliseconds
                'error_rate_high': 5,  # percentage
                'disk_io_high': 1000,  # MB/s
                'network_io_high': 500  # MB/s
            },
            'scaling': {
                'cooldown_minutes': 5,
                'scale_up_threshold': 80,
                'scale_down_threshold': 30,
                'max_instances': 10,
                'min_instances': 1
            }
        }
    
    def _init_clients(self):
        """Initialize various monitoring clients"""
        try:
            # Kubernetes client
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_autoscaling_v1 = client.AutoscalingV1Api()
            self.k8s_metrics = client.CustomObjectsApi()
        except Exception as e:
            logger.warning(f"Kubernetes client initialization failed: {str(e)}")
            self.k8s_apps_v1 = None
        
        try:
            # Redis client for caching metrics
            redis_url = self.config.get('redis', {}).get('url', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Redis client initialization failed: {str(e)}")
            self.redis_client = None
        
        try:
            # AWS CloudWatch client
            self.cloudwatch_client = boto3.client('cloudwatch')
        except Exception as e:
            logger.warning(f"CloudWatch client initialization failed: {str(e)}")
            self.cloudwatch_client = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        interval = self.config['monitoring']['interval_seconds']
        
        while self.monitoring_active:
            try:
                # Collect metrics
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._collect_kubernetes_metrics()
                self._collect_database_metrics()
                
                # Process metrics
                self._process_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Generate recommendations
                if self.config['monitoring']['enable_auto_scaling']:
                    self._generate_scaling_recommendations()
                
                # Clean old data
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(MetricType.CPU_UTILIZATION, cpu_percent, "system")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._add_metric(MetricType.MEMORY_UTILIZATION, memory_percent, "system")
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Calculate throughput (MB/s)
                disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb = disk_io.write_bytes / (1024 * 1024)
                self._add_metric(MetricType.DISK_IO, disk_read_mb + disk_write_mb, "system")
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if network_io:
                network_mb = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
                self._add_metric(MetricType.NETWORK_IO, network_mb, "system")
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # API metrics (if available)
            api_url = self.config.get('api', {}).get('metrics_url', 'http://localhost:8000/metrics')
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    # Parse Prometheus metrics format
                    metrics = self._parse_prometheus_metrics(response.text)
                    for metric_name, value in metrics.items():
                        if 'request_duration' in metric_name:
                            self._add_metric(MetricType.REQUEST_LATENCY, value, "api")
                        elif 'requests_total' in metric_name:
                            self._add_metric(MetricType.THROUGHPUT, value, "api")
                        elif 'error_rate' in metric_name:
                            self._add_metric(MetricType.ERROR_RATE, value, "api")
            except:
                pass  # API metrics not available
            
            # Model inference metrics
            if self.redis_client:
                try:
                    inference_times = self.redis_client.lrange('inference_times', 0, -1)
                    if inference_times:
                        avg_inference_time = np.mean([float(t) for t in inference_times])
                        self._add_metric(MetricType.MODEL_INFERENCE_TIME, avg_inference_time, "model")
                        # Clear the list
                        self.redis_client.delete('inference_times')
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
    
    def _collect_kubernetes_metrics(self):
        """Collect Kubernetes cluster metrics"""
        if not self.k8s_core_v1:
            return
        
        try:
            # Pod metrics
            pods = self.k8s_core_v1.list_pod_for_all_namespaces()
            
            running_pods = 0
            pending_pods = 0
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    running_pods += 1
                elif pod.status.phase == "Pending":
                    pending_pods += 1
            
            self._add_metric(MetricType.THROUGHPUT, running_pods, "kubernetes_pods_running")
            if pending_pods > 0:
                self._add_metric(MetricType.QUEUE_LENGTH, pending_pods, "kubernetes_pods_pending")
            
            # Node metrics
            nodes = self.k8s_core_v1.list_node()
            ready_nodes = sum(1 for node in nodes.items 
                            for condition in node.status.conditions 
                            if condition.type == "Ready" and condition.status == "True")
            
            self._add_metric(MetricType.THROUGHPUT, ready_nodes, "kubernetes_nodes_ready")
            
        except Exception as e:
            logger.error(f"Error collecting Kubernetes metrics: {str(e)}")
    
    def _collect_database_metrics(self):
        """Collect database performance metrics"""
        try:
            # Redis metrics
            if self.redis_client:
                info = self.redis_client.info()
                
                # Memory usage
                used_memory_percent = (info['used_memory'] / info.get('maxmemory', info['used_memory'] * 2)) * 100
                self._add_metric(MetricType.MEMORY_UTILIZATION, used_memory_percent, "redis")
                
                # Cache hit rate
                hits = info.get('keyspace_hits', 0)
                misses = info.get('keyspace_misses', 0)
                if hits + misses > 0:
                    hit_rate = (hits / (hits + misses)) * 100
                    self._add_metric(MetricType.CACHE_HIT_RATE, hit_rate, "redis")
                
                # Connected clients
                connected_clients = info.get('connected_clients', 0)
                self._add_metric(MetricType.THROUGHPUT, connected_clients, "redis_clients")
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {str(e)}")
    
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
                        metrics[metric_name] = metric_value
                except:
                    continue
        return metrics
    
    def _add_metric(self, metric_type: MetricType, value: float, source: str, metadata: Dict[str, Any] = None):
        """Add a metric to the buffer"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata or {}
        )
        
        self.metrics_buffer[metric_type].append(metric)
        
        # Also add to historical data
        self.performance_history[f"{metric_type.value}_{source}"].append({
            'timestamp': metric.timestamp,
            'value': value
        })
    
    def _process_metrics(self):
        """Process collected metrics"""
        try:
            # Calculate moving averages
            for metric_type, metrics in self.metrics_buffer.items():
                if len(metrics) >= 5:  # Need at least 5 data points
                    recent_values = [m.value for m in list(metrics)[-5:]]
                    moving_avg = np.mean(recent_values)
                    
                    # Store processed metric
                    processed_metric = PerformanceMetric(
                        metric_type=metric_type,
                        value=moving_avg,
                        timestamp=datetime.utcnow(),
                        source="processed",
                        metadata={'type': 'moving_average', 'window': 5}
                    )
                    
                    # Add to processed buffer (separate from raw metrics)
                    if not hasattr(self, 'processed_metrics'):
                        self.processed_metrics = defaultdict(lambda: deque(maxlen=100))
                    self.processed_metrics[metric_type].append(processed_metric)
            
            # Detect anomalies
            self._detect_anomalies()
            
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
    
    def _detect_anomalies(self):
        """Detect performance anomalies"""
        try:
            for metric_type, metrics in self.metrics_buffer.items():
                if len(metrics) >= 10:  # Need sufficient data
                    values = [m.value for m in list(metrics)[-10:]]
                    
                    # Simple anomaly detection using standard deviation
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Check if latest value is anomalous (> 2 standard deviations)
                    latest_value = values[-1]
                    if abs(latest_value - mean_val) > 2 * std_val and std_val > 0:
                        # Generate anomaly alert
                        alert = PerformanceAlert(
                            alert_id=f"anomaly_{metric_type.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                            metric_type=metric_type,
                            severity=AlertSeverity.WARNING,
                            message=f"Anomalous {metric_type.value} detected: {latest_value:.2f} (mean: {mean_val:.2f}, std: {std_val:.2f})",
                            threshold=mean_val + 2 * std_val,
                            current_value=latest_value,
                            source="anomaly_detector",
                            timestamp=datetime.utcnow()
                        )
                        self.alerts.append(alert)
                        logger.warning(alert.message)
                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
    
    def _check_alerts(self):
        """Check metrics against thresholds and generate alerts"""
        try:
            current_metrics = {}
            
            # Get latest metrics
            for metric_type, metrics in self.metrics_buffer.items():
                if metrics:
                    current_metrics[metric_type] = metrics[-1].value
            
            # Check thresholds
            for metric_type, value in current_metrics.items():
                threshold_key = f"{metric_type.value}_high"
                if threshold_key in self.thresholds:
                    threshold = self.thresholds[threshold_key]
                    
                    if value > threshold:
                        # Check if we already have an active alert for this
                        existing_alert = any(
                            alert.metric_type == metric_type and not alert.resolved
                            for alert in self.alerts
                        )
                        
                        if not existing_alert:
                            severity = AlertSeverity.CRITICAL if value > threshold * 1.2 else AlertSeverity.WARNING
                            
                            alert = PerformanceAlert(
                                alert_id=f"threshold_{metric_type.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                                metric_type=metric_type,
                                severity=severity,
                                message=f"{metric_type.value} is {value:.2f}, exceeding threshold of {threshold}",
                                threshold=threshold,
                                current_value=value,
                                source="threshold_monitor",
                                timestamp=datetime.utcnow()
                            )
                            self.alerts.append(alert)
                            logger.warning(alert.message)
                            
                            # Send notification
                            self._send_alert_notification(alert)
                
                # Check for resolution of existing alerts
                for alert in self.alerts:
                    if (alert.metric_type == metric_type and 
                        not alert.resolved and 
                        value < alert.threshold * 0.9):  # 10% buffer for resolution
                        
                        alert.resolved = True
                        alert.resolution_time = datetime.utcnow()
                        logger.info(f"Alert resolved: {alert.message}")
                        
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _generate_scaling_recommendations(self):
        """Generate scaling recommendations based on current metrics"""
        try:
            current_metrics = {}
            
            # Get latest processed metrics
            if hasattr(self, 'processed_metrics'):
                for metric_type, metrics in self.processed_metrics.items():
                    if metrics:
                        current_metrics[metric_type] = metrics[-1].value
            
            # CPU-based scaling recommendations
            if MetricType.CPU_UTILIZATION in current_metrics:
                cpu_util = current_metrics[MetricType.CPU_UTILIZATION]
                scale_up_threshold = self.config['scaling']['scale_up_threshold']
                scale_down_threshold = self.config['scaling']['scale_down_threshold']
                
                if cpu_util > scale_up_threshold:
                    recommendation = ScalingRecommendation(
                        recommendation_id=f"scale_up_cpu_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_type="kubernetes_deployment",
                        resource_id="drugban-api",
                        action=ScalingAction.SCALE_OUT,
                        reason=f"High CPU utilization: {cpu_util:.1f}%",
                        current_state={"replicas": self._get_current_replicas("drugban-api")},
                        recommended_state={"replicas": min(self._get_current_replicas("drugban-api") + 1, 
                                                         self.config['scaling']['max_instances'])},
                        estimated_impact={"cpu_reduction": 20, "cost_increase": 50},
                        confidence=0.8,
                        priority=1
                    )
                    self.recommendations.append(recommendation)
                    
                elif cpu_util < scale_down_threshold:
                    current_replicas = self._get_current_replicas("drugban-api")
                    if current_replicas > self.config['scaling']['min_instances']:
                        recommendation = ScalingRecommendation(
                            recommendation_id=f"scale_down_cpu_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                            resource_type="kubernetes_deployment",
                            resource_id="drugban-api",
                            action=ScalingAction.SCALE_IN,
                            reason=f"Low CPU utilization: {cpu_util:.1f}%",
                            current_state={"replicas": current_replicas},
                            recommended_state={"replicas": max(current_replicas - 1,
                                                             self.config['scaling']['min_instances'])},
                            estimated_impact={"cpu_increase": 10, "cost_decrease": 50},
                            confidence=0.7,
                            priority=2
                        )
                        self.recommendations.append(recommendation)
            
            # Memory-based scaling recommendations
            if MetricType.MEMORY_UTILIZATION in current_metrics:
                memory_util = current_metrics[MetricType.MEMORY_UTILIZATION]
                if memory_util > 85:  # High memory usage
                    recommendation = ScalingRecommendation(
                        recommendation_id=f"scale_up_memory_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_type="kubernetes_deployment",
                        resource_id="drugban-api",
                        action=ScalingAction.SCALE_UP,
                        reason=f"High memory utilization: {memory_util:.1f}%",
                        current_state={"memory_limit": "1Gi"},
                        recommended_state={"memory_limit": "2Gi"},
                        estimated_impact={"memory_headroom": 50, "cost_increase": 25},
                        confidence=0.9,
                        priority=1
                    )
                    self.recommendations.append(recommendation)
            
            # Database scaling recommendations
            if MetricType.CACHE_HIT_RATE in current_metrics:
                hit_rate = current_metrics[MetricType.CACHE_HIT_RATE]
                if hit_rate < 80:  # Low cache hit rate
                    recommendation = ScalingRecommendation(
                        recommendation_id=f"optimize_cache_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_type="redis",
                        resource_id="redis-cache",
                        action=ScalingAction.OPTIMIZE,
                        reason=f"Low cache hit rate: {hit_rate:.1f}%",
                        current_state={"maxmemory": "512mb"},
                        recommended_state={"maxmemory": "1gb"},
                        estimated_impact={"hit_rate_increase": 15, "cost_increase": 30},
                        confidence=0.6,
                        priority=3
                    )
                    self.recommendations.append(recommendation)
            
            # Clean old recommendations
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self.recommendations = [
                rec for rec in self.recommendations 
                if any(rec.recommendation_id.endswith(dt.strftime('%Y%m%d%H%M%S')) 
                      and datetime.strptime(rec.recommendation_id.split('_')[-1], '%Y%m%d%H%M%S') > cutoff_time
                      for dt in [datetime.utcnow()])  # This is a simplified check
                or len(rec.recommendation_id.split('_')) < 3
            ]
            
        except Exception as e:
            logger.error(f"Error generating scaling recommendations: {str(e)}")
    
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
        except:
            return 1
    
    def _send_alert_notification(self, alert: PerformanceAlert):
        """Send alert notification"""
        try:
            # Slack notification
            slack_webhook = self.config.get('notifications', {}).get('slack_webhook')
            if slack_webhook:
                color = {
                    AlertSeverity.INFO: "good",
                    AlertSeverity.WARNING: "warning", 
                    AlertSeverity.CRITICAL: "danger",
                    AlertSeverity.EMERGENCY: "#8B0000"
                }.get(alert.severity, "warning")
                
                payload = {
                    "attachments": [{
                        "color": color,
                        "title": f"Performance Alert - {alert.severity.value.upper()}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Metric", "value": alert.metric_type.value, "short": True},
                            {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                            {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
                            {"title": "Source", "value": alert.source, "short": True}
                        ],
                        "timestamp": int(alert.timestamp.timestamp())
                    }]
                }
                
                requests.post(slack_webhook, json=payload, timeout=5)
                
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            retention_hours = self.config['monitoring']['retention_hours']
            cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
            
            # Clean metrics buffer
            for metric_type in list(self.metrics_buffer.keys()):
                metrics = self.metrics_buffer[metric_type]
                # Keep only recent metrics
                recent_metrics = deque([m for m in metrics if m.timestamp > cutoff_time], maxlen=1000)
                self.metrics_buffer[metric_type] = recent_metrics
            
            # Clean processed metrics
            if hasattr(self, 'processed_metrics'):
                for metric_type in list(self.processed_metrics.keys()):
                    metrics = self.processed_metrics[metric_type]
                    recent_metrics = deque([m for m in metrics if m.timestamp > cutoff_time], maxlen=100)
                    self.processed_metrics[metric_type] = recent_metrics
            
            # Clean alerts (keep resolved alerts for 24 hours)
            alert_cutoff = datetime.utcnow() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts
                if not alert.resolved or alert.resolution_time > alert_cutoff
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    def apply_scaling_recommendation(self, recommendation_id: str, dry_run: bool = True) -> bool:
        """Apply a scaling recommendation"""
        recommendation = None
        for rec in self.recommendations:
            if rec.recommendation_id == recommendation_id:
                recommendation = rec
                break
        
        if not recommendation:
            logger.error(f"Recommendation {recommendation_id} not found")
            return False
        
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would apply recommendation: {recommendation.reason}")
                return True
            
            if recommendation.resource_type == "kubernetes_deployment":
                return self._apply_k8s_scaling(recommendation)
            elif recommendation.resource_type == "redis":
                return self._apply_redis_optimization(recommendation)
            else:
                logger.warning(f"Unknown resource type: {recommendation.resource_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying scaling recommendation: {str(e)}")
            return False
    
    def _apply_k8s_scaling(self, recommendation: ScalingRecommendation) -> bool:
        """Apply Kubernetes scaling recommendation"""
        if not self.k8s_apps_v1:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            if recommendation.action == ScalingAction.SCALE_OUT:
                # Scale out (increase replicas)
                new_replicas = recommendation.recommended_state["replicas"]
                
                self.k8s_apps_v1.patch_namespaced_deployment_scale(
                    name=recommendation.resource_id,
                    namespace="drugban",
                    body={"spec": {"replicas": new_replicas}}
                )
                
                logger.info(f"Scaled out {recommendation.resource_id} to {new_replicas} replicas")
                return True
                
            elif recommendation.action == ScalingAction.SCALE_IN:
                # Scale in (decrease replicas)
                new_replicas = recommendation.recommended_state["replicas"]
                
                self.k8s_apps_v1.patch_namespaced_deployment_scale(
                    name=recommendation.resource_id,
                    namespace="drugban",
                    body={"spec": {"replicas": new_replicas}}
                )
                
                logger.info(f"Scaled in {recommendation.resource_id} to {new_replicas} replicas")
                return True
                
            elif recommendation.action == ScalingAction.SCALE_UP:
                # Scale up (increase resources)
                memory_limit = recommendation.recommended_state["memory_limit"]
                
                # This would require more complex patch operation
                logger.info(f"Would scale up memory to {memory_limit} for {recommendation.resource_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error applying Kubernetes scaling: {str(e)}")
            return False
    
    def _apply_redis_optimization(self, recommendation: ScalingRecommendation) -> bool:
        """Apply Redis optimization recommendation"""
        if not self.redis_client:
            logger.error("Redis client not available")
            return False
        
        try:
            if recommendation.action == ScalingAction.OPTIMIZE:
                # Configure Redis memory settings
                maxmemory = recommendation.recommended_state["maxmemory"]
                
                # This would typically be done through Redis configuration
                # For now, just log the recommendation
                logger.info(f"Would optimize Redis with maxmemory={maxmemory}")
                return True
                
        except Exception as e:
            logger.error(f"Error applying Redis optimization: {str(e)}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_active': self.monitoring_active,
            'current_metrics': {},
            'active_alerts': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Current metrics
        for metric_type, metrics in self.metrics_buffer.items():
            if metrics:
                latest = metrics[-1]
                summary['current_metrics'][metric_type.value] = {
                    'value': latest.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'source': latest.source
                }
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        summary['active_alerts'] = [
            {
                'alert_id': alert.alert_id,
                'metric_type': alert.metric_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in active_alerts
        ]
        
        # Recent recommendations
        recent_recommendations = sorted(
            self.recommendations, 
            key=lambda x: x.priority, 
            reverse=True
        )[:5]
        
        summary['recommendations'] = [
            {
                'recommendation_id': rec.recommendation_id,
                'action': rec.action.value,
                'reason': rec.reason,
                'confidence': rec.confidence,
                'priority': rec.priority,
                'estimated_impact': rec.estimated_impact
            }
            for rec in recent_recommendations
        ]
        
        # Statistics
        summary['statistics'] = {
            'total_metrics_collected': sum(len(metrics) for metrics in self.metrics_buffer.values()),
            'total_alerts_generated': len(self.alerts),
            'active_alerts_count': len(active_alerts),
            'recommendations_count': len(self.recommendations)
        }
        
        return summary
    
    def generate_performance_report(self, output_path: str = "performance_report.html"):
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary()
        
        # Create visualizations
        self._create_performance_visualizations()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background-color: #ffe4e1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .recommendation {{ background-color: #f0fff0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .critical {{ border-left: 5px solid #dc3545; }}
                .warning {{ border-left: 5px solid #ffc107; }}
                .info {{ border-left: 5px solid #17a2b8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Performance Monitoring Report</h1>
            <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <h2>Current System Status</h2>
            <div class="metric">
                <h3>Monitoring Status: {"Active" if summary['monitoring_active'] else "Inactive"}</h3>
                <p>Total Metrics Collected: {summary['statistics']['total_metrics_collected']}</p>
                <p>Active Alerts: {summary['statistics']['active_alerts_count']}</p>
                <p>Pending Recommendations: {summary['statistics']['recommendations_count']}</p>
            </div>
            
            <h2>Current Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th>Last Updated</th>
                </tr>
        """
        
        for metric_name, metric_data in summary['current_metrics'].items():
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td>{metric_data['value']:.2f}</td>
                    <td>{metric_data['source']}</td>
                    <td>{metric_data['timestamp']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Active Alerts</h2>
        """
        
        if summary['active_alerts']:
            for alert in summary['active_alerts']:
                severity_class = alert['severity']
                html_content += f"""
                <div class="alert {severity_class}">
                    <h3>{alert['severity'].upper()}: {alert['metric_type'].replace('_', ' ').title()}</h3>
                    <p>{alert['message']}</p>
                    <p><small>Alert ID: {alert['alert_id']}</small></p>
                </div>
                """
        else:
            html_content += "<p>No active alerts</p>"
        
        html_content += """
            <h2>Scaling Recommendations</h2>
        """
        
        if summary['recommendations']:
            for rec in summary['recommendations']:
                html_content += f"""
                <div class="recommendation">
                    <h3>{rec['action'].replace('_', ' ').title()}</h3>
                    <p><strong>Reason:</strong> {rec['reason']}</p>
                    <p><strong>Confidence:</strong> {rec['confidence']:.1%}</p>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Estimated Impact:</strong> {rec['estimated_impact']}</p>
                    <p><small>ID: {rec['recommendation_id']}</small></p>
                </div>
                """
        else:
            html_content += "<p>No recommendations at this time</p>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_path}")
    
    def _create_performance_visualizations(self):
        """Create performance visualization charts"""
        try:
            # Create charts directory
            charts_dir = Path("charts")
            charts_dir.mkdir(exist_ok=True)
            
            # CPU utilization over time
            if MetricType.CPU_UTILIZATION in self.metrics_buffer:
                cpu_metrics = self.metrics_buffer[MetricType.CPU_UTILIZATION]
                if len(cpu_metrics) > 1:
                    timestamps = [m.timestamp for m in cpu_metrics]
                    values = [m.value for m in cpu_metrics]
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(timestamps, values, label='CPU Utilization %')
                    plt.axhline(y=self.thresholds.get('cpu_utilization_high', 80), 
                               color='r', linestyle='--', label='High Threshold')
                    plt.xlabel('Time')
                    plt.ylabel('CPU Utilization (%)')
                    plt.title('CPU Utilization Over Time')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(charts_dir / 'cpu_utilization.png', dpi=300)
                    plt.close()
            
            # Memory utilization over time
            if MetricType.MEMORY_UTILIZATION in self.metrics_buffer:
                memory_metrics = self.metrics_buffer[MetricType.MEMORY_UTILIZATION]
                if len(memory_metrics) > 1:
                    timestamps = [m.timestamp for m in memory_metrics]
                    values = [m.value for m in memory_metrics]
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(timestamps, values, label='Memory Utilization %', color='orange')
                    plt.axhline(y=self.thresholds.get('memory_utilization_high', 85), 
                               color='r', linestyle='--', label='High Threshold')
                    plt.xlabel('Time')
                    plt.ylabel('Memory Utilization (%)')
                    plt.title('Memory Utilization Over Time')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(charts_dir / 'memory_utilization.png', dpi=300)
                    plt.close()
            
            logger.info("Performance visualizations created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")


def main():
    """Example usage of performance monitor"""
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Let it run for a bit
        time.sleep(60)
        
        # Get summary
        summary = monitor.get_performance_summary()
        print("\nPerformance Summary:")
        print(f"Current Metrics: {len(summary['current_metrics'])}")
        print(f"Active Alerts: {len(summary['active_alerts'])}")
        print(f"Recommendations: {len(summary['recommendations'])}")
        
        # Generate report
        monitor.generate_performance_report()
        
        # Apply recommendations (dry run)
        for rec in summary['recommendations']:
            success = monitor.apply_scaling_recommendation(rec['recommendation_id'], dry_run=True)
            print(f"Recommendation {rec['recommendation_id']}: {'Success' if success else 'Failed'}")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()