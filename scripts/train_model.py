#!/usr/bin/env python3
"""
Model Training Script with PyTorch Lightning and MLflow Integration

This script implements the training pipeline for drug-target interaction prediction
with experiment tracking, checkpointing, and model versioning.
"""

import os
import sys
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

class DrugTargetDataset(Dataset):
    """Dataset for drug-target interaction prediction."""
    
    def __init__(self, features_dir, split='train', augmented=False):
        self.features_dir = Path(features_dir)
        self.split = split
        self.augmented = augmented
        
        # Load features
        self.load_features()
        
        # Load labels
        self.load_labels()
        
    def load_features(self):
        """Load molecular and target features."""
        if self.augmented and self.split == 'train':
            feature_path = self.features_dir / f"{self.split}_augmented"
        else:
            feature_path = self.features_dir / self.split
            
        # Load different feature types
        self.morgan_features = np.load(feature_path / "morgan_features.npy")
        self.descriptor_features = np.load(feature_path / "descriptors_features.npy")
        self.target_features = np.load(feature_path / "target_features.npy")
        self.interaction_features = np.load(feature_path / "interaction_features.npy")
        
        logger.info(f"Loaded features for {self.split}: {len(self.morgan_features)} samples")
        
    def load_labels(self):
        """Load interaction labels."""
        # Load corresponding dataset file
        label_file = Path("data/labeled") / f"{self.split}_dataset.json"
        
        if self.augmented and self.split == 'train':
            label_file = Path("data/labeled") / f"{self.split}_augmented_dataset.json"
            
        with open(label_file, 'r') as f:
            data = json.load(f)
            
        # Extract labels (convert interaction type to binary)
        self.labels = []
        for entry in data:
            interaction_type = entry.get('interaction_type', 'unknown')
            # Binary classification: agonist=1, antagonist=0
            label = 1 if interaction_type == 'agonist' else 0
            self.labels.append(label)
            
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Combine features
        drug_features = np.concatenate([
            self.morgan_features[idx],
            self.descriptor_features[idx]
        ])
        
        target_features = self.target_features[idx]
        
        # Return as tensors
        return {
            'drug_features': torch.FloatTensor(drug_features),
            'target_features': torch.FloatTensor(target_features),
            'label': torch.FloatTensor([self.labels[idx]])
        }

class DrugTargetInteractionModel(pl.LightningModule):
    """PyTorch Lightning model for drug-target interaction prediction."""
    
    def __init__(self, drug_dim, target_dim, hidden_dim=256, dropout=0.3, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction prediction head
        interaction_dim = hidden_dim // 2 + hidden_dim // 4
        self.interaction_head = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.loss_fn = nn.BCELoss()
        
    def forward(self, drug_features, target_features):
        # Encode drug and target
        drug_encoded = self.drug_encoder(drug_features)
        target_encoded = self.target_encoder(target_features)
        
        # Concatenate representations
        combined = torch.cat([drug_encoded, target_encoded], dim=1)
        
        # Predict interaction
        interaction_prob = self.interaction_head(combined)
        
        return interaction_prob
    
    def training_step(self, batch, batch_idx):
        drug_features = batch['drug_features']
        target_features = batch['target_features']
        labels = batch['label']
        
        # Forward pass
        predictions = self(drug_features, target_features)
        loss = self.loss_fn(predictions, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log accuracy
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == labels).float().mean()
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        drug_features = batch['drug_features']
        target_features = batch['target_features']
        labels = batch['label']
        
        # Forward pass
        predictions = self(drug_features, target_features)
        loss = self.loss_fn(predictions, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate metrics
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == labels).float().mean()
        
        # Store predictions for epoch-level metrics
        return {
            'loss': loss,
            'predictions': predictions.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'accuracy': accuracy
        }
    
    def validation_epoch_end(self, outputs):
        # Aggregate predictions
        all_predictions = np.concatenate([x['predictions'] for x in outputs])
        all_labels = np.concatenate([x['labels'] for x in outputs])
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_predictions)
            self.log('val_auc', auc, prog_bar=True)
        except:
            self.log('val_auc', 0.5, prog_bar=True)
        
        # Calculate average accuracy
        avg_accuracy = np.mean([x['accuracy'].item() for x in outputs])
        self.log('val_accuracy', avg_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }

def create_data_loaders(params, augmented=True):
    """Create PyTorch data loaders."""
    features_dir = Path("data/processed/features")
    
    # Create datasets
    train_dataset = DrugTargetDataset(features_dir, split='train', augmented=augmented)
    val_dataset = DrugTargetDataset(features_dir, split='validation')
    test_dataset = DrugTargetDataset(features_dir, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['training']['batch_size'],
        shuffle=True,
        num_workers=params['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['training']['batch_size'],
        shuffle=False,
        num_workers=params['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['training']['batch_size'],
        shuffle=False,
        num_workers=params['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_model(params, experiment_name="drug_repurposing_training"):
    """Train the drug-target interaction model."""
    logger.info("Starting model training...")
    
    # Set random seed
    seed_everything(params['training']['random_seed'])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(params, augmented=True)
    
    # Calculate feature dimensions
    sample_batch = next(iter(train_loader))
    drug_dim = sample_batch['drug_features'].shape[1]
    target_dim = sample_batch['target_features'].shape[1]
    
    logger.info(f"Feature dimensions - Drug: {drug_dim}, Target: {target_dim}")
    
    # Initialize model
    model = DrugTargetInteractionModel(
        drug_dim=drug_dim,
        target_dim=target_dim,
        hidden_dim=params['model']['hidden_dim'],
        dropout=params['model']['dropout'],
        lr=params['training']['learning_rate']
    )
    
    # Set up MLflow
    mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Create MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("models/checkpoints"),
        filename="drug_target_model-{epoch:02d}-{val_loss:.2f}",
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=params['training']['early_stopping_patience'],
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = Trainer(
        max_epochs=params['training']['max_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=params['training']['gradient_clip_val'],
        accumulate_grad_batches=params['training']['accumulate_grad_batches'],
        log_every_n_steps=10,
        val_check_interval=1.0,
        deterministic=True
    )
    
    # Log parameters to MLflow
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params({
            "model_type": "DrugTargetInteractionModel",
            "drug_dim": drug_dim,
            "target_dim": target_dim,
            "hidden_dim": params['model']['hidden_dim'],
            "dropout": params['model']['dropout'],
            "learning_rate": params['training']['learning_rate'],
            "batch_size": params['training']['batch_size'],
            "max_epochs": params['training']['max_epochs'],
            "augmented_data": True,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset)
        })
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Test model
        test_results = trainer.test(model, test_loader)
        
        # Log test results
        if test_results:
            mlflow.log_metrics({
                "test_loss": test_results[0].get('test_loss', 0),
                "test_accuracy": test_results[0].get('test_accuracy', 0)
            })
        
        # Save model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="DrugTargetInteractionModel"
        )
        
        # Save best checkpoint path
        mlflow.log_artifact(checkpoint_callback.best_model_path)
        
    logger.info(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")
    
    return model, trainer

def main():
    """Main training function."""
    logger.info("🚀 Starting Drug-Target Interaction Model Training")
    
    try:
        # Load parameters
        params = load_params()
        
        # Train model
        model, trainer = train_model(params)
        
        print("\n✅ Model Training Complete!")
        print("=" * 50)
        print("📊 Access MLflow UI at: http://127.0.0.1:5000")
        print("🔍 View experiments in: drug_repurposing_training")
        print("💾 Model checkpoints saved in: models/checkpoints/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())