#!/usr/bin/env python3
"""
Data and Concept Drift Monitoring for DrugBAN Model
Uses Evidently AI for comprehensive drift detection and analysis
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from collections import deque
import pickle
import os

# Evidently AI imports
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import (
    DataDriftTable, DatasetDriftMetric, DatasetMissingValuesMetric,
    DatasetCorrelationsMetric, ColumnDriftMetric, ColumnSummaryMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns, TestNumberOfDuplicatedRows,
    TestColumnsType, TestNumberOfDuplicatedColumns, TestDataDrift,
    TestTargetDrift, TestValueRange, TestMeanInNSigmas
)

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, AllChem

# Monitoring imports
from .metrics import DrugPredictionMetrics, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    timestamp: float
    drift_detected: bool
    drift_score: float
    drift_method: str
    feature_type: str
    details: Dict[str, Any]
    p_value: Optional[float] = None
    threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConceptDriftResult:
    """Result of concept drift analysis"""
    timestamp: float
    drift_detected: bool
    drift_score: float
    drift_method: str
    model_performance_change: float
    statistical_significance: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MolecularFeatureExtractor:
    """Extract molecular features for drift analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)  # Reduce dimensionality for drift detection
        self.fitted = False
        
    def extract_molecular_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """Extract RDKit molecular descriptors"""
        descriptors = []
        descriptor_names = [name for name, _ in Descriptors._descList]
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    desc_values = [Descriptors._descList[i][1](mol) for i in range(len(Descriptors._descList))]
                    # Replace NaN values with 0
                    desc_values = [0 if pd.isna(val) else val for val in desc_values]
                    descriptors.append(desc_values)
                else:
                    # Invalid SMILES, use zeros
                    descriptors.append([0] * len(descriptor_names))
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                descriptors.append([0] * len(descriptor_names))
        
        return np.array(descriptors)
    
    def extract_morgan_fingerprints(self, smiles_list: List[str], 
                                   radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Extract Morgan fingerprints"""
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fingerprints.append(np.array(fp))
                else:
                    fingerprints.append(np.zeros(n_bits))
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                fingerprints.append(np.zeros(n_bits))
        
        return np.array(fingerprints)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and PCA on reference data"""
        if features.shape[1] > 50:  # Only apply PCA if more than 50 features
            scaled_features = self.scaler.fit_transform(features)
            reduced_features = self.pca.fit_transform(scaled_features)
        else:
            reduced_features = self.scaler.fit_transform(features)
        
        self.fitted = True
        return reduced_features
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler and PCA"""
        if not self.fitted:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        if features.shape[1] > 50 and hasattr(self.pca, 'components_'):
            scaled_features = self.scaler.transform(features)
            reduced_features = self.pca.transform(scaled_features)
        else:
            reduced_features = self.scaler.transform(features)
        
        return reduced_features

class DataDriftDetector:
    """
    Comprehensive data drift detection using multiple methods
    """
    
    def __init__(self, reference_data: pd.DataFrame, 
                 column_mapping: Optional[ColumnMapping] = None):
        self.reference_data = reference_data
        self.column_mapping = column_mapping or ColumnMapping()
        self.feature_extractor = MolecularFeatureExtractor()
        self.drift_history = deque(maxlen=1000)
        
        # Fit feature extractor on reference data
        if 'smiles' in reference_data.columns:
            smiles_features = self.feature_extractor.extract_molecular_descriptors(
                reference_data['smiles'].tolist()
            )
            self.reference_features = self.feature_extractor.fit_transform(smiles_features)
        
    def detect_evidently_drift(self, current_data: pd.DataFrame) -> DriftDetectionResult:
        """Detect drift using Evidently AI"""
        try:
            # Create data drift report
            data_drift_report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
                DatasetMissingValuesMetric(),
                DatasetCorrelationsMetric()
            ])
            
            data_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract drift metrics
            report_dict = data_drift_report.as_dict()
            dataset_drift = report_dict['metrics'][0]['result']
            
            drift_score = dataset_drift.get('drift_share', 0.0)
            drift_detected = dataset_drift.get('dataset_drift', False)
            
            # Detailed analysis
            drift_details = {
                'number_of_columns': dataset_drift.get('number_of_columns', 0),
                'number_of_drifted_columns': dataset_drift.get('number_of_drifted_columns', 0),
                'share_of_drifted_columns': dataset_drift.get('share_of_drifted_columns', 0.0),
                'dataset_drift': drift_detected
            }
            
            # Add column-level drift information
            if len(report_dict['metrics']) > 1:
                column_drift_info = report_dict['metrics'][1]['result']
                drift_details['column_drift'] = column_drift_info
            
            result = DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=drift_detected,
                drift_score=drift_score,
                drift_method="evidently_ai",
                feature_type="tabular",
                details=drift_details
            )
            
            self.drift_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in Evidently drift detection: {e}")
            return DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=False,
                drift_score=0.0,
                drift_method="evidently_ai",
                feature_type="tabular",
                details={"error": str(e)}
            )
    
    def detect_molecular_drift(self, current_smiles: List[str]) -> DriftDetectionResult:
        """Detect drift in molecular features"""
        try:
            # Extract features from current data
            current_features = self.feature_extractor.extract_molecular_descriptors(current_smiles)
            current_transformed = self.feature_extractor.transform(current_features)
            
            # Statistical tests for drift
            drift_scores = []
            p_values = []
            
            for i in range(min(current_transformed.shape[1], self.reference_features.shape[1])):
                ref_feature = self.reference_features[:, i]
                curr_feature = current_transformed[:, i]
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
                drift_scores.append(ks_stat)
                p_values.append(p_value)
            
            # Overall drift score (mean KS statistic)
            overall_drift_score = np.mean(drift_scores)
            min_p_value = np.min(p_values) if p_values else 1.0
            
            # Drift threshold
            drift_threshold = 0.1
            drift_detected = overall_drift_score > drift_threshold or min_p_value < 0.05
            
            result = DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                drift_method="kolmogorov_smirnov",
                feature_type="molecular_descriptors",
                details={
                    "ks_statistics": drift_scores,
                    "p_values": p_values,
                    "min_p_value": min_p_value,
                    "num_features": len(drift_scores)
                },
                p_value=min_p_value,
                threshold=drift_threshold
            )
            
            self.drift_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in molecular drift detection: {e}")
            return DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=False,
                drift_score=0.0,
                drift_method="kolmogorov_smirnov",
                feature_type="molecular_descriptors",
                details={"error": str(e)}
            )
    
    def detect_fingerprint_drift(self, current_smiles: List[str]) -> DriftDetectionResult:
        """Detect drift in molecular fingerprints"""
        try:
            # Extract fingerprints
            ref_smiles = self.reference_data['smiles'].tolist() if 'smiles' in self.reference_data.columns else []
            ref_fingerprints = self.feature_extractor.extract_morgan_fingerprints(ref_smiles)
            curr_fingerprints = self.feature_extractor.extract_morgan_fingerprints(current_smiles)
            
            # Calculate Tanimoto similarity distributions
            ref_similarities = []
            curr_similarities = []
            
            # Sample comparisons to avoid quadratic complexity
            sample_size = min(100, len(ref_fingerprints), len(curr_fingerprints))
            ref_sample = ref_fingerprints[:sample_size]
            curr_sample = curr_fingerprints[:sample_size]
            
            # Calculate pairwise similarities within each set
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    ref_sim = np.sum(ref_sample[i] & ref_sample[j]) / np.sum(ref_sample[i] | ref_sample[j])
                    curr_sim = np.sum(curr_sample[i] & curr_sample[j]) / np.sum(curr_sample[i] | curr_sample[j])
                    
                    ref_similarities.append(ref_sim)
                    curr_similarities.append(curr_sim)
            
            # Statistical comparison
            if ref_similarities and curr_similarities:
                ks_stat, p_value = stats.ks_2samp(ref_similarities, curr_similarities)
                drift_detected = p_value < 0.05
            else:
                ks_stat, p_value = 0.0, 1.0
                drift_detected = False
            
            result = DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=drift_detected,
                drift_score=ks_stat,
                drift_method="fingerprint_similarity",
                feature_type="morgan_fingerprints",
                details={
                    "reference_similarities": ref_similarities[:10],  # Store sample
                    "current_similarities": curr_similarities[:10],
                    "sample_size": sample_size
                },
                p_value=p_value
            )
            
            self.drift_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in fingerprint drift detection: {e}")
            return DriftDetectionResult(
                timestamp=time.time(),
                drift_detected=False,
                drift_score=0.0,
                drift_method="fingerprint_similarity",
                feature_type="morgan_fingerprints",
                details={"error": str(e)}
            )

class ConceptDriftDetector:
    """
    Concept drift detection for model performance changes
    """
    
    def __init__(self, reference_performance: Dict[str, float]):
        self.reference_performance = reference_performance
        self.performance_history = deque(maxlen=1000)
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.fitted = False
        
    def detect_performance_drift(self, current_predictions: List[PredictionResult]) -> ConceptDriftResult:
        """Detect concept drift based on model performance changes"""
        try:
            # Filter predictions with ground truth
            labeled_predictions = [p for p in current_predictions if p.true_label is not None]
            
            if len(labeled_predictions) < 10:
                return ConceptDriftResult(
                    timestamp=time.time(),
                    drift_detected=False,
                    drift_score=0.0,
                    drift_method="performance_comparison",
                    model_performance_change=0.0,
                    statistical_significance=1.0,
                    details={"error": "Insufficient labeled data"}
                )
            
            # Calculate current performance
            y_true = [p.true_label for p in labeled_predictions]
            y_pred = [1 if p.prediction > 0.5 else 0 for p in labeled_predictions]
            y_score = [p.prediction for p in labeled_predictions]
            
            current_performance = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Calculate performance changes
            performance_changes = {}
            for metric, current_value in current_performance.items():
                reference_value = self.reference_performance.get(metric, current_value)
                change = abs(current_value - reference_value) / max(reference_value, 0.001)
                performance_changes[metric] = change
            
            # Overall performance change score
            overall_change = np.mean(list(performance_changes.values()))
            
            # Store performance history
            self.performance_history.append(current_performance)
            
            # Statistical significance test
            if len(self.performance_history) >= 20:
                # Fit anomaly detector on performance history
                history_array = np.array([[p['accuracy'], p['precision'], p['recall'], p['f1_score']] 
                                        for p in list(self.performance_history)])
                
                if not self.fitted:
                    self.anomaly_detector.fit(history_array)
                    self.fitted = True
                
                # Check if current performance is anomalous
                current_array = np.array([[current_performance['accuracy'], 
                                         current_performance['precision'],
                                         current_performance['recall'], 
                                         current_performance['f1_score']]])
                
                anomaly_score = self.anomaly_detector.decision_function(current_array)[0]
                is_anomaly = self.anomaly_detector.predict(current_array)[0] == -1
                
                statistical_significance = 1.0 - stats.norm.cdf(abs(anomaly_score))
            else:
                statistical_significance = 1.0
                is_anomaly = False
            
            # Drift detection logic
            drift_threshold = 0.2  # 20% performance change
            drift_detected = overall_change > drift_threshold or is_anomaly
            
            result = ConceptDriftResult(
                timestamp=time.time(),
                drift_detected=drift_detected,
                drift_score=overall_change,
                drift_method="performance_comparison",
                model_performance_change=overall_change,
                statistical_significance=statistical_significance,
                details={
                    "current_performance": current_performance,
                    "reference_performance": self.reference_performance,
                    "performance_changes": performance_changes,
                    "is_anomaly": is_anomaly,
                    "sample_size": len(labeled_predictions)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
            return ConceptDriftResult(
                timestamp=time.time(),
                drift_detected=False,
                drift_score=0.0,
                drift_method="performance_comparison",
                model_performance_change=0.0,
                statistical_significance=1.0,
                details={"error": str(e)}
            )

class DriftMonitor:
    """
    Main drift monitoring system combining data and concept drift detection
    """
    
    def __init__(self, reference_data: pd.DataFrame, 
                 reference_performance: Dict[str, float],
                 metrics: DrugPredictionMetrics):
        self.reference_data = reference_data
        self.reference_performance = reference_performance
        self.metrics = metrics
        
        # Initialize detectors
        column_mapping = ColumnMapping(
            target=None,
            prediction=None,
            numerical_features=[col for col in reference_data.columns 
                              if reference_data[col].dtype in ['int64', 'float64']],
            categorical_features=[col for col in reference_data.columns 
                                if reference_data[col].dtype == 'object' and col != 'smiles']
        )
        
        self.data_drift_detector = DataDriftDetector(reference_data, column_mapping)
        self.concept_drift_detector = ConceptDriftDetector(reference_performance)
        
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self, check_interval: int = 300):  # 5 minutes
        """Start continuous drift monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Drift monitoring started")
    
    def stop_monitoring(self):
        """Stop drift monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Drift monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get recent predictions (placeholder - would connect to actual data)
                recent_predictions = self._get_recent_predictions()
                
                if recent_predictions:
                    # Run drift detection
                    drift_results = self.check_all_drift_types(recent_predictions)
                    
                    # Update metrics
                    for result in drift_results:
                        if isinstance(result, DriftDetectionResult):
                            self.metrics.record_data_drift(
                                result.feature_type,
                                result.drift_method,
                                result.drift_score
                            )
                        elif isinstance(result, ConceptDriftResult):
                            self.metrics.record_concept_drift(
                                result.drift_method,
                                result.drift_score
                            )
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _get_recent_predictions(self) -> List[PredictionResult]:
        """Get recent predictions for drift analysis (placeholder)"""
        # This would be implemented to retrieve recent predictions from the system
        # For now, return empty list
        return []
    
    def check_all_drift_types(self, predictions: List[PredictionResult]) -> List:
        """Run all drift detection methods"""
        results = []
        
        try:
            # Prepare current data
            current_data = self._prepare_current_data(predictions)
            
            if not current_data.empty:
                # Data drift detection
                evidently_result = self.data_drift_detector.detect_evidently_drift(current_data)
                results.append(evidently_result)
                
                # Molecular drift detection
                if 'smiles' in current_data.columns:
                    smiles_list = current_data['smiles'].tolist()
                    
                    molecular_result = self.data_drift_detector.detect_molecular_drift(smiles_list)
                    results.append(molecular_result)
                    
                    fingerprint_result = self.data_drift_detector.detect_fingerprint_drift(smiles_list)
                    results.append(fingerprint_result)
            
            # Concept drift detection
            concept_result = self.concept_drift_detector.detect_performance_drift(predictions)
            results.append(concept_result)
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
        
        return results
    
    def _prepare_current_data(self, predictions: List[PredictionResult]) -> pd.DataFrame:
        """Convert predictions to DataFrame for drift analysis"""
        data = []
        
        for pred in predictions:
            # Extract basic features (would be enhanced with actual feature extraction)
            row = {
                'drug_id': pred.drug_id,
                'target_id': pred.target_id,
                'prediction': pred.prediction,
                'confidence': pred.confidence,
                'timestamp': pred.timestamp
            }
            
            # Add SMILES if available (placeholder)
            # row['smiles'] = get_smiles_for_drug(pred.drug_id)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        data_drift_history = list(self.data_drift_detector.drift_history)
        concept_drift_history = list(self.concept_drift_detector.performance_history)
        
        summary = {
            "data_drift": {
                "total_checks": len(data_drift_history),
                "recent_drift_detected": any(r.drift_detected for r in data_drift_history[-10:]),
                "avg_drift_score": np.mean([r.drift_score for r in data_drift_history]) if data_drift_history else 0.0
            },
            "concept_drift": {
                "performance_history_length": len(concept_drift_history),
                "current_performance": concept_drift_history[-1] if concept_drift_history else None,
                "reference_performance": self.reference_performance
            },
            "monitoring_status": {
                "active": self.monitoring_active,
                "last_check": datetime.now().isoformat()
            }
        }
        
        return summary

# Utility functions for integration
def create_drift_monitor(reference_data_path: str, 
                        reference_performance: Dict[str, float],
                        metrics: DrugPredictionMetrics) -> DriftMonitor:
    """Create and configure drift monitor"""
    try:
        reference_data = pd.read_csv(reference_data_path)
        return DriftMonitor(reference_data, reference_performance, metrics)
    except Exception as e:
        logger.error(f"Error creating drift monitor: {e}")
        raise

def save_drift_results(results: List, output_path: str):
    """Save drift detection results to file"""
    serializable_results = []
    for result in results:
        if hasattr(result, 'to_dict'):
            serializable_results.append(result.to_dict())
        else:
            serializable_results.append(str(result))
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

if __name__ == "__main__":
    # Example usage
    from .metrics import get_metrics_instance
    
    # Create sample reference data
    reference_data = pd.DataFrame({
        'drug_id': [f'DRUG_{i}' for i in range(100)],
        'target_id': [f'TARGET_{i%10}' for i in range(100)],
        'prediction': np.random.random(100),
        'confidence': np.random.random(100)
    })
    
    reference_performance = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.79,
        'f1_score': 0.80
    }
    
    # Initialize drift monitor
    metrics = get_metrics_instance()
    monitor = DriftMonitor(reference_data, reference_performance, metrics)
    
    # Start monitoring
    monitor.start_monitoring(check_interval=60)  # Check every minute
    
    try:
        # Keep running
        time.sleep(300)  # Run for 5 minutes
    finally:
        monitor.stop_monitoring()
    
    # Print summary
    summary = monitor.get_drift_summary()
    print("Drift Monitoring Summary:")
    print(json.dumps(summary, indent=2))