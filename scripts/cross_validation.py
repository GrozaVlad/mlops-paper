#!/usr/bin/env python3
"""
Cross-Validation Framework for Drug-Target Interaction Models

This script implements k-fold cross-validation with stratification
for robust model evaluation and performance estimation.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from train_model import DrugTargetInteractionModel, DrugTargetDataset, load_params

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossValidationFramework:
    """Framework for k-fold cross-validation of drug-target models."""
    
    def __init__(self, params, n_folds=5, random_state=42):
        self.params = params
        self.n_folds = n_folds
        self.random_state = random_state
        seed_everything(random_state)
        
        # Results storage
        self.fold_results = []
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        
    def prepare_cv_splits(self, dataset):
        """Prepare cross-validation splits with stratification."""
        logger.info(f"Preparing {self.n_folds}-fold cross-validation splits...")
        
        # Get all labels for stratification
        all_labels = []
        for i in range(len(dataset)):
            all_labels.append(dataset[i]['label'].item())
        all_labels = np.array(all_labels)
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            splits.append({
                'fold': fold_idx,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            
            # Log class distribution
            train_labels = all_labels[train_idx]
            val_labels = all_labels[val_idx]
            logger.info(f"Fold {fold_idx}: Train size={len(train_idx)}, Val size={len(val_idx)}")
            logger.info(f"  Train class distribution: {np.bincount(train_labels.astype(int))}")
            logger.info(f"  Val class distribution: {np.bincount(val_labels.astype(int))}")
            
        return splits, all_labels
    
    def train_fold(self, dataset, fold_info, fold_idx):
        """Train model on a single fold."""
        logger.info(f"Training fold {fold_idx}/{self.n_folds-1}...")
        
        # Create subset datasets
        train_subset = Subset(dataset, fold_info['train_indices'])
        val_subset = Subset(dataset, fold_info['val_indices'])
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.params['training']['batch_size'],
            shuffle=True,
            num_workers=self.params['training']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.params['training']['batch_size'],
            shuffle=False,
            num_workers=self.params['training']['num_workers'],
            pin_memory=True
        )
        
        # Get feature dimensions
        sample_batch = next(iter(train_loader))
        drug_dim = sample_batch['drug_features'].shape[1]
        target_dim = sample_batch['target_features'].shape[1]
        
        # Initialize model
        model = DrugTargetInteractionModel(
            drug_dim=drug_dim,
            target_dim=target_dim,
            hidden_dim=self.params['model']['hidden_dim'],
            dropout=self.params['model']['dropout'],
            lr=self.params['training']['learning_rate']
        )
        
        # Set up MLflow logger
        mlflow_logger = MLFlowLogger(
            experiment_name="drug_repurposing_cross_validation",
            tracking_uri=mlflow.get_tracking_uri(),
            run_name=f"cv_fold_{fold_idx}"
        )
        
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(f"models/cv/fold_{fold_idx}"),
            filename="best_model",
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.params['training']['early_stopping_patience'],
            mode='min'
        )
        
        # Create trainer
        trainer = Trainer(
            max_epochs=self.params['training']['max_epochs'],
            gpus=1 if torch.cuda.is_available() else 0,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping],
            gradient_clip_val=self.params['training']['gradient_clip_val'],
            log_every_n_steps=10,
            enable_progress_bar=True,
            deterministic=True
        )
        
        # Log fold info to MLflow
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_params({
                'cv_fold': fold_idx,
                'cv_total_folds': self.n_folds,
                'fold_train_size': fold_info['train_size'],
                'fold_val_size': fold_info['val_size']
            })
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Get best model metrics
            best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
            best_val_auc = trainer.callback_metrics.get('val_auc', 0.0)
            best_val_accuracy = trainer.callback_metrics.get('val_accuracy', 0.0)
            
            mlflow.log_metrics({
                'fold_best_val_loss': best_val_loss.item() if hasattr(best_val_loss, 'item') else best_val_loss,
                'fold_best_val_auc': best_val_auc.item() if hasattr(best_val_auc, 'item') else best_val_auc,
                'fold_best_val_accuracy': best_val_accuracy.item() if hasattr(best_val_accuracy, 'item') else best_val_accuracy
            })
        
        # Evaluate on validation set
        val_predictions, val_labels, val_probabilities = self.evaluate_model(
            model, val_loader, checkpoint_callback.best_model_path
        )
        
        # Calculate fold metrics
        fold_metrics = self.calculate_metrics(val_labels, val_predictions, val_probabilities)
        fold_metrics['fold'] = fold_idx
        fold_metrics['best_model_path'] = checkpoint_callback.best_model_path
        
        return fold_metrics, val_predictions, val_labels, val_probabilities
    
    def evaluate_model(self, model, data_loader, checkpoint_path):
        """Evaluate model on a dataset."""
        # Load best checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                drug_feat = batch['drug_features'].to(device)
                target_feat = batch['target_features'].to(device)
                batch_labels = batch['label'].to(device)
                
                outputs = model(drug_feat, target_feat)
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                predictions.extend(preds.flatten())
                labels.extend(batch_labels.cpu().numpy().flatten())
                probabilities.extend(probs.flatten())
        
        return np.array(predictions), np.array(labels), np.array(probabilities)
    
    def calculate_metrics(self, labels, predictions, probabilities):
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(predictions == labels)
        metrics['precision'] = np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        metrics['recall'] = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0
        metrics['f1'] = f1_score(labels, predictions)
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(labels, probabilities)
        except:
            metrics['auc'] = 0.5
            
        # Confusion matrix elements
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tp = np.sum((predictions == 1) & (labels == 1))
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def run_cross_validation(self):
        """Run complete cross-validation."""
        logger.info(f"Starting {self.n_folds}-fold cross-validation...")
        
        # Load full dataset
        features_dir = Path("data/processed/features")
        
        # Combine train and validation for CV
        train_dataset = DrugTargetDataset(features_dir, split='train', augmented=True)
        val_dataset = DrugTargetDataset(features_dir, split='validation')
        
        # Combine datasets
        combined_features = {
            'morgan': np.vstack([train_dataset.morgan_features, val_dataset.morgan_features]),
            'descriptors': np.vstack([train_dataset.descriptor_features, val_dataset.descriptor_features]),
            'target': np.vstack([train_dataset.target_features, val_dataset.target_features]),
            'labels': np.concatenate([train_dataset.labels, val_dataset.labels])
        }
        
        # Create combined dataset
        class CombinedDataset(Dataset):
            def __init__(self, features_dict):
                self.morgan_features = features_dict['morgan']
                self.descriptor_features = features_dict['descriptors']
                self.target_features = features_dict['target']
                self.labels = features_dict['labels']
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                drug_features = np.concatenate([
                    self.morgan_features[idx],
                    self.descriptor_features[idx]
                ])
                return {
                    'drug_features': torch.FloatTensor(drug_features),
                    'target_features': torch.FloatTensor(self.target_features[idx]),
                    'label': torch.FloatTensor([self.labels[idx]])
                }
        
        combined_dataset = CombinedDataset(combined_features)
        
        # Prepare CV splits
        splits, all_labels = self.prepare_cv_splits(combined_dataset)
        
        # Set up MLflow
        mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
        mlflow.set_experiment("drug_repurposing_cross_validation")
        
        # Run cross-validation
        with mlflow.start_run(run_name=f"{self.n_folds}_fold_cv_summary"):
            mlflow.log_params({
                'cv_folds': self.n_folds,
                'cv_random_state': self.random_state,
                'total_samples': len(combined_dataset),
                'model_type': 'DrugTargetInteractionModel'
            })
            
            for fold_idx, fold_info in enumerate(splits):
                fold_metrics, predictions, labels, probabilities = self.train_fold(
                    combined_dataset, fold_info, fold_idx
                )
                
                self.fold_results.append(fold_metrics)
                self.all_predictions.extend(predictions)
                self.all_labels.extend(labels)
                self.all_probabilities.extend(probabilities)
                
                logger.info(f"Fold {fold_idx} metrics: {fold_metrics}")
            
            # Calculate aggregate metrics
            aggregate_metrics = self.calculate_aggregate_metrics()
            
            # Log aggregate metrics to MLflow
            for metric_name, metric_value in aggregate_metrics.items():
                if isinstance(metric_value, dict):
                    for sub_name, sub_value in metric_value.items():
                        mlflow.log_metric(f"{metric_name}_{sub_name}", sub_value)
                else:
                    mlflow.log_metric(metric_name, metric_value)
            
            # Create visualizations
            self.create_cv_visualizations()
            
            # Save results
            results = {
                'cv_configuration': {
                    'n_folds': self.n_folds,
                    'random_state': self.random_state,
                    'total_samples': len(combined_dataset)
                },
                'fold_results': self.fold_results,
                'aggregate_metrics': aggregate_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = Path("models/cv/cv_results.json")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            mlflow.log_artifact(results_file)
            
        return aggregate_metrics
    
    def calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all folds."""
        logger.info("Calculating aggregate cross-validation metrics...")
        
        # Convert to numpy arrays
        fold_metrics_df = pd.DataFrame(self.fold_results)
        
        # Calculate mean and std for each metric
        aggregate_metrics = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity']:
            values = fold_metrics_df[metric].values
            aggregate_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values.tolist()
            }
        
        # Overall performance
        aggregate_metrics['cv_mean_accuracy'] = aggregate_metrics['accuracy']['mean']
        aggregate_metrics['cv_std_accuracy'] = aggregate_metrics['accuracy']['std']
        aggregate_metrics['cv_mean_auc'] = aggregate_metrics['auc']['mean']
        aggregate_metrics['cv_std_auc'] = aggregate_metrics['auc']['std']
        
        return aggregate_metrics
    
    def create_cv_visualizations(self):
        """Create cross-validation visualizations."""
        logger.info("Creating cross-validation visualizations...")
        
        output_dir = Path("models/cv/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Fold performance comparison
        fold_metrics_df = pd.DataFrame(self.fold_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_to_plot = ['accuracy', 'auc', 'f1', 'sensitivity']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = fold_metrics_df[metric].values
            folds = fold_metrics_df['fold'].values
            
            ax.bar(folds, values, alpha=0.7, color='skyblue', edgecolor='navy')
            ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(values):.3f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} by Fold')
            ax.legend()
            ax.set_xticks(folds)
            
        plt.tight_layout()
        plt.savefig(output_dir / "fold_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metric distribution boxplot
        metrics_data = []
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            for value in fold_metrics_df[metric].values:
                metrics_data.append({'Metric': metric.capitalize(), 'Value': value})
                
        metrics_df = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df, x='Metric', y='Value')
        plt.title('Cross-Validation Metric Distributions')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / "cv_metric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(str(output_dir))
        
        logger.info(f"Visualizations saved to: {output_dir}")

def main():
    """Main cross-validation function."""
    logger.info("🔄 Starting Cross-Validation Framework")
    
    try:
        # Load parameters
        params = load_params()
        
        # Initialize cross-validation framework
        cv_framework = CrossValidationFramework(
            params=params,
            n_folds=params['evaluation']['cross_validation_folds'],
            random_state=params['training']['random_seed']
        )
        
        # Run cross-validation
        aggregate_metrics = cv_framework.run_cross_validation()
        
        print("\n✅ Cross-Validation Complete!")
        print("=" * 60)
        print(f"📊 {params['evaluation']['cross_validation_folds']}-Fold Cross-Validation Results:")
        print(f"  Accuracy: {aggregate_metrics['accuracy']['mean']:.3f} ± {aggregate_metrics['accuracy']['std']:.3f}")
        print(f"  AUC:      {aggregate_metrics['auc']['mean']:.3f} ± {aggregate_metrics['auc']['std']:.3f}")
        print(f"  F1 Score: {aggregate_metrics['f1']['mean']:.3f} ± {aggregate_metrics['f1']['std']:.3f}")
        print(f"  Precision:{aggregate_metrics['precision']['mean']:.3f} ± {aggregate_metrics['precision']['std']:.3f}")
        print(f"  Recall:   {aggregate_metrics['recall']['mean']:.3f} ± {aggregate_metrics['recall']['std']:.3f}")
        print("\n📁 Results saved to: models/cv/")
        print("📊 Access MLflow UI at: http://127.0.0.1:5000")
        print("🔍 View experiment: drug_repurposing_cross_validation")
        
        return 0
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())