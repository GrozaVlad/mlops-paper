#!/usr/bin/env python3
"""
Error Analysis Framework for Drug-Target Interaction Models

This script provides comprehensive error analysis including confusion matrices,
compound-level analysis, and model interpretability.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE

import mlflow
import mlflow.pytorch

# SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# LIME for model interpretability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

from train_model import DrugTargetDataset, DrugTargetInteractionModel, load_params

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelErrorAnalyzer:
    """Comprehensive error analysis for drug-target interaction models."""
    
    def __init__(self, model, test_loader, device='cpu'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store predictions and labels
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.drug_features = []
        self.target_features = []
        
    def collect_predictions(self):
        """Collect all predictions from the test set."""
        logger.info("Collecting predictions from test set...")
        
        with torch.no_grad():
            for batch in self.test_loader:
                drug_feat = batch['drug_features'].to(self.device)
                target_feat = batch['target_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions
                outputs = self.model(drug_feat, target_feat)
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                self.predictions.extend(preds.flatten())
                self.labels.extend(labels.cpu().numpy().flatten())
                self.probabilities.extend(probs.flatten())
                self.drug_features.extend(drug_feat.cpu().numpy())
                self.target_features.extend(target_feat.cpu().numpy())
        
        self.predictions = np.array(self.predictions)
        self.labels = np.array(self.labels)
        self.probabilities = np.array(self.probabilities)
        self.drug_features = np.array(self.drug_features)
        self.target_features = np.array(self.target_features)
        
        logger.info(f"Collected {len(self.predictions)} predictions")
        
    def create_confusion_matrix(self, save_path=None):
        """Create and visualize confusion matrix."""
        logger.info("Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.labels, self.predictions)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Antagonist', 'Agonist'],
                   yticklabels=['Antagonist', 'Agonist'])
        plt.title('Confusion Matrix - Drug-Target Interactions')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        plt.text(0.5, -0.15, f'Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | Precision: {precision:.3f}',
                ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
            
        return cm, {'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision}
    
    def analyze_error_patterns(self):
        """Analyze patterns in prediction errors."""
        logger.info("Analyzing error patterns...")
        
        # Identify errors
        errors = self.predictions != self.labels
        error_indices = np.where(errors)[0]
        
        # Analyze false positives and false negatives
        false_positives = (self.predictions == 1) & (self.labels == 0)
        false_negatives = (self.predictions == 0) & (self.labels == 1)
        
        fp_indices = np.where(false_positives)[0]
        fn_indices = np.where(false_negatives)[0]
        
        # Analyze confidence in errors
        fp_confidences = self.probabilities[fp_indices]
        fn_confidences = self.probabilities[fn_indices]
        
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(self.labels),
            'false_positives': {
                'count': len(fp_indices),
                'mean_confidence': np.mean(fp_confidences) if len(fp_confidences) > 0 else 0,
                'std_confidence': np.std(fp_confidences) if len(fp_confidences) > 0 else 0
            },
            'false_negatives': {
                'count': len(fn_indices),
                'mean_confidence': np.mean(fn_confidences) if len(fn_confidences) > 0 else 0,
                'std_confidence': np.std(fn_confidences) if len(fn_confidences) > 0 else 0
            }
        }
        
        return error_analysis, error_indices
    
    def create_roc_curve(self, save_path=None):
        """Create ROC curve visualization."""
        logger.info("Creating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Drug-Target Interaction Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")
            
        return roc_auc, {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
    
    def visualize_feature_space(self, save_path=None):
        """Visualize feature space using t-SNE."""
        logger.info("Creating feature space visualization...")
        
        # Combine drug and target features
        combined_features = np.concatenate([self.drug_features, self.target_features], axis=1)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_features)-1))
        features_2d = tsne.fit_transform(combined_features)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot correct predictions
        correct = self.predictions == self.labels
        scatter1 = plt.scatter(features_2d[correct, 0], features_2d[correct, 1], 
                              c=self.labels[correct], cmap='viridis', alpha=0.6, 
                              label='Correct', marker='o')
        
        # Plot errors
        errors = ~correct
        scatter2 = plt.scatter(features_2d[errors, 0], features_2d[errors, 1], 
                              c=self.labels[errors], cmap='viridis', alpha=0.9, 
                              label='Error', marker='X', s=100, edgecolors='red', linewidth=2)
        
        plt.colorbar(scatter1, label='True Label (0: Antagonist, 1: Agonist)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Feature Space Visualization (t-SNE)')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature space visualization saved to: {save_path}")
            
        return features_2d
    
    def explain_predictions_shap(self, num_samples=100):
        """Use SHAP to explain model predictions."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explanation")
            return None
            
        logger.info(f"Creating SHAP explanations for {num_samples} samples...")
        
        # Create a wrapper function for SHAP
        def model_predict(features):
            drug_feat = torch.FloatTensor(features[:, :self.drug_features.shape[1]])
            target_feat = torch.FloatTensor(features[:, self.drug_features.shape[1]:])
            with torch.no_grad():
                outputs = self.model(drug_feat.to(self.device), target_feat.to(self.device))
            return outputs.cpu().numpy()
        
        # Combine features
        combined_features = np.concatenate([self.drug_features, self.target_features], axis=1)
        
        # Create SHAP explainer
        sample_indices = np.random.choice(len(combined_features), min(num_samples, len(combined_features)), replace=False)
        background = combined_features[sample_indices]
        
        explainer = shap.KernelExplainer(model_predict, background)
        
        # Calculate SHAP values for a subset
        shap_sample_indices = np.random.choice(len(combined_features), min(20, len(combined_features)), replace=False)
        shap_values = explainer.shap_values(combined_features[shap_sample_indices])
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, combined_features[shap_sample_indices], 
                         show=False, plot_type="bar")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("models/error_analysis/shap_importance.png", dpi=300, bbox_inches='tight')
        
        return shap_values
    
    def explain_predictions_lime(self, num_samples=10):
        """Use LIME to explain individual predictions."""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available, skipping explanation")
            return None
            
        logger.info(f"Creating LIME explanations for {num_samples} samples...")
        
        # Combine features
        combined_features = np.concatenate([self.drug_features, self.target_features], axis=1)
        
        # Create feature names
        drug_feature_names = [f"drug_feat_{i}" for i in range(self.drug_features.shape[1])]
        target_feature_names = [f"target_feat_{i}" for i in range(self.target_features.shape[1])]
        feature_names = drug_feature_names + target_feature_names
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            combined_features,
            feature_names=feature_names,
            class_names=['Antagonist', 'Agonist'],
            mode='classification'
        )
        
        # Explain random samples
        explanations = []
        sample_indices = np.random.choice(len(combined_features), min(num_samples, len(combined_features)), replace=False)
        
        for idx in sample_indices:
            # Create prediction function
            def predict_fn(features):
                drug_feat = torch.FloatTensor(features[:, :self.drug_features.shape[1]])
                target_feat = torch.FloatTensor(features[:, self.drug_features.shape[1]:])
                with torch.no_grad():
                    outputs = self.model(drug_feat.to(self.device), target_feat.to(self.device))
                probs = outputs.cpu().numpy()
                return np.column_stack([1 - probs, probs])
            
            # Get explanation
            exp = explainer.explain_instance(
                combined_features[idx],
                predict_fn,
                num_features=20
            )
            
            explanations.append({
                'index': idx,
                'true_label': self.labels[idx],
                'predicted_label': self.predictions[idx],
                'probability': self.probabilities[idx],
                'explanation': exp.as_list()
            })
            
            # Save explanation figure
            fig = exp.as_pyplot_figure()
            plt.title(f"LIME Explanation - Sample {idx}")
            plt.tight_layout()
            plt.savefig(f"models/error_analysis/lime_sample_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        return explanations

def run_error_analysis(model_path=None, experiment_name="drug_repurposing_training"):
    """Run comprehensive error analysis."""
    logger.info("Starting error analysis...")
    
    # Load parameters
    params = load_params()
    
    # Create output directory
    output_dir = Path("models/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    features_dir = Path("data/processed/features")
    test_dataset = DrugTargetDataset(features_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    if model_path is None:
        # Find latest model from MLflow
        mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Get latest run
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from run: {run_id}")
        else:
            logger.error("No trained models found")
            return None
    else:
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model = DrugTargetInteractionModel.load_from_checkpoint(model_path)
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create analyzer
    analyzer = ModelErrorAnalyzer(model, test_loader, device)
    
    # Collect predictions
    analyzer.collect_predictions()
    
    # Run analyses
    results = {}
    
    # 1. Confusion Matrix
    cm, cm_metrics = analyzer.create_confusion_matrix(output_dir / "confusion_matrix.png")
    results['confusion_matrix'] = cm.tolist()
    results['confusion_matrix_metrics'] = cm_metrics
    
    # 2. Error Patterns
    error_analysis, error_indices = analyzer.analyze_error_patterns()
    results['error_analysis'] = error_analysis
    
    # 3. ROC Curve
    roc_auc, roc_data = analyzer.create_roc_curve(output_dir / "roc_curve.png")
    results['roc_auc'] = roc_auc
    
    # 4. Feature Space Visualization
    features_2d = analyzer.visualize_feature_space(output_dir / "feature_space.png")
    
    # 5. Model Interpretability
    if SHAP_AVAILABLE:
        shap_values = analyzer.explain_predictions_shap()
        
    if LIME_AVAILABLE:
        lime_explanations = analyzer.explain_predictions_lime()
        results['lime_explanations'] = lime_explanations
    
    # 6. Classification Report
    report = classification_report(analyzer.labels, analyzer.predictions, 
                                 target_names=['Antagonist', 'Agonist'],
                                 output_dict=True)
    results['classification_report'] = report
    
    # Save results
    results['analysis_timestamp'] = datetime.now().isoformat()
    results['model_path'] = str(model_path) if model_path else model_uri
    results['test_samples'] = len(analyzer.labels)
    
    with open(output_dir / "error_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log to MLflow
    with mlflow.start_run(run_name="error_analysis"):
        mlflow.log_artifacts(str(output_dir))
        mlflow.log_metrics({
            'test_accuracy': 1 - error_analysis['error_rate'],
            'test_roc_auc': roc_auc,
            'test_sensitivity': cm_metrics['sensitivity'],
            'test_specificity': cm_metrics['specificity'],
            'test_precision': cm_metrics['precision']
        })
        
    logger.info(f"Error analysis complete. Results saved to: {output_dir}")
    
    return results

def main():
    """Main error analysis function."""
    logger.info("🔍 Starting Comprehensive Error Analysis")
    
    try:
        # Run error analysis
        results = run_error_analysis()
        
        if results:
            print("\n✅ Error Analysis Complete!")
            print("=" * 60)
            print(f"📊 Test Accuracy: {1 - results['error_analysis']['error_rate']:.3f}")
            print(f"📈 ROC AUC: {results['roc_auc']:.3f}")
            print(f"🎯 Sensitivity: {results['confusion_matrix_metrics']['sensitivity']:.3f}")
            print(f"🎯 Specificity: {results['confusion_matrix_metrics']['specificity']:.3f}")
            print(f"🎯 Precision: {results['confusion_matrix_metrics']['precision']:.3f}")
            print("\n📁 Results saved to: models/error_analysis/")
            print("📊 Visualizations:")
            print("  - Confusion Matrix: confusion_matrix.png")
            print("  - ROC Curve: roc_curve.png")
            print("  - Feature Space: feature_space.png")
            if SHAP_AVAILABLE:
                print("  - SHAP Analysis: shap_importance.png")
            if LIME_AVAILABLE:
                print("  - LIME Explanations: lime_sample_*.png")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())