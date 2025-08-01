#!/usr/bin/env python3
"""
Custom Metrics for DrugBAN Model Monitoring
Implements Prometheus metrics for drug prediction quality and system performance
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Enum, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import threading
from collections import deque, defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structure for tracking prediction results"""
    drug_id: str
    target_id: str
    prediction: float
    confidence: float
    true_label: Optional[int] = None
    timestamp: float = None
    processing_time: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class DrugPredictionMetrics:
    """
    Custom Prometheus metrics for DrugBAN model monitoring
    Tracks prediction quality, model performance, and system health
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        self._prediction_history = deque(maxlen=10000)  # Store last 10k predictions
        self._model_performance_cache = {}
        self._drift_metrics = {}
        self._lock = threading.Lock()
        
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # Prediction metrics
        self.prediction_total = Counter(
            'drugban_predictions_total',
            'Total number of drug-target predictions made',
            ['drug_class', 'target_class', 'prediction_type'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'drugban_prediction_latency_seconds',
            'Time taken to make a prediction',
            ['prediction_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'drugban_prediction_confidence',
            'Distribution of prediction confidence scores',
            ['drug_class', 'target_class'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'drugban_model_accuracy',
            'Current model accuracy (rolling window)',
            ['window_size'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'drugban_model_precision',
            'Current model precision (rolling window)',
            ['class', 'window_size'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'drugban_model_recall',
            'Current model recall (rolling window)',
            ['class', 'window_size'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'drugban_model_f1_score',
            'Current model F1 score (rolling window)',
            ['class', 'window_size'],
            registry=self.registry
        )
        
        self.model_auc = Gauge(
            'drugban_model_auc',
            'Current model AUC score (rolling window)',
            ['window_size'],
            registry=self.registry
        )
        
        # Data drift metrics
        self.data_drift_score = Gauge(
            'drugban_data_drift_score',
            'Data drift detection score',
            ['feature_type', 'drift_method'],
            registry=self.registry
        )
        
        self.concept_drift_score = Gauge(
            'drugban_concept_drift_score',
            'Concept drift detection score',
            ['drift_method'],
            registry=self.registry
        )
        
        # System health metrics
        self.model_load_time = Gauge(
            'drugban_model_load_time_seconds',
            'Time taken to load the model',
            registry=self.registry
        )
        
        self.feature_extraction_time = Histogram(
            'drugban_feature_extraction_time_seconds',
            'Time taken for feature extraction',
            ['feature_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'drugban_model_memory_usage_bytes',
            'Memory usage of the model',
            registry=self.registry
        )
        
        # Business metrics
        self.successful_predictions = Counter(
            'drugban_successful_predictions_total',
            'Number of successful predictions',
            ['drug_class', 'target_class'],
            registry=self.registry
        )
        
        self.failed_predictions = Counter(
            'drugban_failed_predictions_total',
            'Number of failed predictions',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.high_confidence_predictions = Counter(
            'drugban_high_confidence_predictions_total',
            'Number of high confidence predictions (>0.8)',
            ['drug_class', 'target_class'],
            registry=self.registry
        )
        
        # Model version and status
        self.model_version = Gauge(
            'drugban_model_version',
            'Current model version',
            ['version', 'model_type'],
            registry=self.registry
        )
        
        self.model_status = Enum(
            'drugban_model_status',
            'Current model status',
            states=['healthy', 'degraded', 'failed'],
            registry=self.registry
        )
        
        # Alert metrics
        self.alert_triggers = Counter(
            'drugban_alert_triggers_total',
            'Number of alerts triggered',
            ['alert_type', 'severity'],
            registry=self.registry
        )
    
    def record_prediction(self, result: PredictionResult, 
                         drug_class: str = "unknown", 
                         target_class: str = "unknown"):
        """Record a prediction result and update metrics"""
        
        with self._lock:
            # Store prediction in history
            self._prediction_history.append(result)
            
            # Update counters
            prediction_type = "batch" if hasattr(result, 'batch_size') else "single"
            self.prediction_total.labels(
                drug_class=drug_class,
                target_class=target_class,
                prediction_type=prediction_type
            ).inc()
            
            # Record latency
            if result.processing_time:
                self.prediction_latency.labels(
                    prediction_type=prediction_type
                ).observe(result.processing_time)
            
            # Record confidence
            self.prediction_confidence.labels(
                drug_class=drug_class,
                target_class=target_class
            ).observe(result.confidence)
            
            # Update successful predictions
            self.successful_predictions.labels(
                drug_class=drug_class,
                target_class=target_class
            ).inc()
            
            # High confidence tracking
            if result.confidence > 0.8:
                self.high_confidence_predictions.labels(
                    drug_class=drug_class,
                    target_class=target_class
                ).inc()
        
        # Update rolling window metrics
        self._update_rolling_metrics()
    
    def record_prediction_error(self, error_type: str, component: str):
        """Record a prediction error"""
        self.failed_predictions.labels(
            error_type=error_type,
            component=component
        ).inc()
        
        # Update model status if too many errors
        self._check_model_health()
    
    def record_feature_extraction_time(self, feature_type: str, duration: float):
        """Record feature extraction timing"""
        self.feature_extraction_time.labels(
            feature_type=feature_type
        ).observe(duration)
    
    def update_model_info(self, version: str, model_type: str, load_time: float):
        """Update model version and load time"""
        self.model_version.labels(
            version=version,
            model_type=model_type
        ).set(1)
        
        self.model_load_time.set(load_time)
    
    def update_memory_usage(self, memory_bytes: int):
        """Update model memory usage"""
        self.model_memory_usage.set(memory_bytes)
    
    def record_data_drift(self, feature_type: str, drift_method: str, score: float):
        """Record data drift detection score"""
        self.data_drift_score.labels(
            feature_type=feature_type,
            drift_method=drift_method
        ).set(score)
        
        # Check for drift alerts
        if score > 0.7:  # Threshold for drift alert
            self.trigger_alert("data_drift", "warning")
        elif score > 0.9:
            self.trigger_alert("data_drift", "critical")
    
    def record_concept_drift(self, drift_method: str, score: float):
        """Record concept drift detection score"""
        self.concept_drift_score.labels(
            drift_method=drift_method
        ).set(score)
        
        # Check for concept drift alerts
        if score > 0.6:
            self.trigger_alert("concept_drift", "warning")
        elif score > 0.8:
            self.trigger_alert("concept_drift", "critical")
    
    def trigger_alert(self, alert_type: str, severity: str):
        """Trigger an alert"""
        self.alert_triggers.labels(
            alert_type=alert_type,
            severity=severity
        ).inc()
        
        logger.warning(f"Alert triggered: {alert_type} - {severity}")
    
    def _update_rolling_metrics(self):
        """Update rolling window performance metrics"""
        if len(self._prediction_history) < 10:
            return
        
        # Get predictions with ground truth
        recent_predictions = list(self._prediction_history)[-1000:]  # Last 1000
        predictions_with_truth = [p for p in recent_predictions if p.true_label is not None]
        
        if len(predictions_with_truth) < 10:
            return
        
        # Calculate metrics
        y_true = [p.true_label for p in predictions_with_truth]
        y_pred = [1 if p.prediction > 0.5 else 0 for p in predictions_with_truth]
        y_score = [p.prediction for p in predictions_with_truth]
        
        # Update metrics for different window sizes
        for window_size in [100, 500, 1000]:
            if len(predictions_with_truth) >= window_size:
                window_true = y_true[-window_size:]
                window_pred = y_pred[-window_size:]
                window_score = y_score[-window_size:]
                
                # Accuracy
                accuracy = accuracy_score(window_true, window_pred)
                self.model_accuracy.labels(window_size=str(window_size)).set(accuracy)
                
                # Precision and Recall per class
                precision = precision_score(window_true, window_pred, average=None, zero_division=0)
                recall = recall_score(window_true, window_pred, average=None, zero_division=0)
                f1 = f1_score(window_true, window_pred, average=None, zero_division=0)
                
                for class_idx, (p, r, f) in enumerate(zip(precision, recall, f1)):
                    class_name = str(class_idx)
                    self.model_precision.labels(class=class_name, window_size=str(window_size)).set(p)
                    self.model_recall.labels(class=class_name, window_size=str(window_size)).set(r)
                    self.model_f1_score.labels(class=class_name, window_size=str(window_size)).set(f)
                
                # AUC
                if len(set(window_true)) > 1:  # Need both classes for AUC
                    auc = roc_auc_score(window_true, window_score)
                    self.model_auc.labels(window_size=str(window_size)).set(auc)
    
    def _check_model_health(self):
        """Check model health and update status"""
        recent_predictions = list(self._prediction_history)[-100:]
        
        if not recent_predictions:
            return
        
        # Calculate error rate
        total_recent = len(recent_predictions)
        failed_recent = sum(1 for p in recent_predictions if p.prediction is None)
        error_rate = failed_recent / total_recent if total_recent > 0 else 0
        
        # Calculate performance degradation
        predictions_with_truth = [p for p in recent_predictions if p.true_label is not None]
        
        if len(predictions_with_truth) >= 20:
            y_true = [p.true_label for p in predictions_with_truth]
            y_pred = [1 if p.prediction > 0.5 else 0 for p in predictions_with_truth]
            accuracy = accuracy_score(y_true, y_pred)
            
            if error_rate > 0.1 or accuracy < 0.6:
                self.model_status.state('failed')
                self.trigger_alert("model_health", "critical")
            elif error_rate > 0.05 or accuracy < 0.7:
                self.model_status.state('degraded')
                self.trigger_alert("model_health", "warning")
            else:
                self.model_status.state('healthy')
        else:
            if error_rate > 0.1:
                self.model_status.state('failed')
            elif error_rate > 0.05:
                self.model_status.state('degraded')
            else:
                self.model_status.state('healthy')
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of current metrics"""
        recent_predictions = list(self._prediction_history)[-100:]
        
        if not recent_predictions:
            return {"status": "no_data"}
        
        predictions_with_truth = [p for p in recent_predictions if p.true_label is not None]
        
        summary = {
            "total_predictions": len(self._prediction_history),
            "recent_predictions": len(recent_predictions),
            "predictions_with_ground_truth": len(predictions_with_truth),
            "avg_confidence": np.mean([p.confidence for p in recent_predictions]),
            "avg_processing_time": np.mean([p.processing_time for p in recent_predictions if p.processing_time])
        }
        
        if predictions_with_truth:
            y_true = [p.true_label for p in predictions_with_truth]
            y_pred = [1 if p.prediction > 0.5 else 0 for p in predictions_with_truth]
            
            summary.update({
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            })
        
        return summary
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def start_metrics_server(self, port: int = 8001):
        """Start metrics server for Prometheus scraping"""
        start_http_server(port, registry=self.registry)
        logger.info(f"Metrics server started on port {port}")

class ModelPerformanceMonitor:
    """
    Advanced model performance monitoring with sliding windows and alerts
    """
    
    def __init__(self, metrics: DrugPredictionMetrics):
        self.metrics = metrics
        self.performance_history = defaultdict(deque)
        self.alert_thresholds = {
            'accuracy': 0.75,
            'precision': 0.70,
            'recall': 0.70,
            'f1_score': 0.70,
            'auc': 0.75
        }
        
    def monitor_batch_performance(self, predictions: List[PredictionResult]):
        """Monitor performance for a batch of predictions"""
        
        # Filter predictions with ground truth
        labeled_predictions = [p for p in predictions if p.true_label is not None]
        
        if len(labeled_predictions) < 10:
            return
        
        # Calculate batch metrics
        y_true = [p.true_label for p in labeled_predictions]
        y_pred = [1 if p.prediction > 0.5 else 0 for p in labeled_predictions]
        y_score = [p.prediction for p in labeled_predictions]
        
        batch_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'timestamp': time.time()
        }
        
        if len(set(y_true)) > 1:
            batch_metrics['auc'] = roc_auc_score(y_true, y_score)
        
        # Store in history
        for metric, value in batch_metrics.items():
            if metric != 'timestamp':
                self.performance_history[metric].append((batch_metrics['timestamp'], value))
                
                # Keep only last 100 measurements
                if len(self.performance_history[metric]) > 100:
                    self.performance_history[metric].popleft()
        
        # Check for performance degradation
        self._check_performance_alerts()
    
    def _check_performance_alerts(self):
        """Check for performance degradation and trigger alerts"""
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in self.performance_history:
                recent_values = [v for _, v in list(self.performance_history[metric])[-10:]]
                
                if len(recent_values) >= 5:
                    avg_recent = np.mean(recent_values)
                    
                    if avg_recent < threshold:
                        self.metrics.trigger_alert(f"performance_{metric}", "warning")
                        logger.warning(f"Performance alert: {metric} = {avg_recent:.3f} < {threshold}")

# Global metrics instance
global_metrics = DrugPredictionMetrics()

def get_metrics_instance() -> DrugPredictionMetrics:
    """Get the global metrics instance"""
    return global_metrics

if __name__ == "__main__":
    # Example usage and testing
    metrics = DrugPredictionMetrics()
    
    # Simulate some predictions
    for i in range(100):
        result = PredictionResult(
            drug_id=f"DRUG_{i}",
            target_id=f"TARGET_{i % 10}",
            prediction=np.random.random(),
            confidence=np.random.random(),
            true_label=np.random.randint(0, 2),
            processing_time=np.random.uniform(0.01, 0.1)
        )
        
        drug_class = "small_molecule" if i % 2 == 0 else "protein"
        target_class = "GPCR" if i % 3 == 0 else "kinase"
        
        metrics.record_prediction(result, drug_class, target_class)
    
    # Print metrics summary
    print("Metrics Summary:")
    print(json.dumps(metrics.get_metrics_summary(), indent=2))
    
    # Start metrics server (comment out for testing)
    # metrics.start_metrics_server(8001)