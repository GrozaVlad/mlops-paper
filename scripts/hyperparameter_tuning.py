#!/usr/bin/env python3
"""
Hyperparameter Tuning Script with Optuna and MLflow

This script performs automated hyperparameter optimization for drug-target
interaction models using Optuna with MLflow tracking.
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import mlflow
import mlflow.pytorch

from train_model import DrugTargetDataset, DrugTargetInteractionModel, create_data_loaders

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

class OptimizableModel(DrugTargetInteractionModel):
    """Extended model class for hyperparameter optimization."""
    
    def __init__(self, trial, drug_dim, target_dim):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        # Additional architecture choices
        n_layers = trial.suggest_int("n_layers", 2, 4)
        activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "ELU"])
        
        super().__init__(drug_dim, target_dim, hidden_dim, dropout, lr)
        
        # Store trial for later use
        self.trial = trial
        self.hparams["n_layers"] = n_layers
        self.hparams["activation"] = activation
        
        # Rebuild architecture based on suggestions
        self._rebuild_architecture()
        
    def _rebuild_architecture(self):
        """Rebuild model architecture based on hyperparameters."""
        hidden_dim = self.hparams.hidden_dim
        dropout = self.hparams.dropout
        n_layers = self.hparams.n_layers
        
        # Get activation function
        if self.hparams.activation == "ReLU":
            activation = nn.ReLU
        elif self.hparams.activation == "LeakyReLU":
            activation = nn.LeakyReLU
        else:
            activation = nn.ELU
            
        # Build drug encoder with variable depth
        drug_layers = []
        input_dim = self.hparams.drug_dim
        
        for i in range(n_layers):
            output_dim = hidden_dim // (2 ** i)
            drug_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            input_dim = output_dim
            
        self.drug_encoder = nn.Sequential(*drug_layers)
        
        # Update interaction head input dimension
        final_drug_dim = hidden_dim // (2 ** (n_layers - 1))
        target_dim = hidden_dim // 4
        
        self.interaction_head = nn.Sequential(
            nn.Linear(final_drug_dim + target_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def validation_epoch_end(self, outputs):
        """Override to report metrics to Optuna."""
        super().validation_epoch_end(outputs)
        
        # Get current metrics
        val_loss = self.trainer.callback_metrics.get("val_loss", float('inf'))
        val_auc = self.trainer.callback_metrics.get("val_auc", 0.0)
        
        # Report to Optuna (we want to maximize AUC)
        self.trial.report(val_auc, self.current_epoch)
        
        # Check if should prune
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def objective(trial, params):
    """Optuna objective function for hyperparameter optimization."""
    
    # Set random seed
    seed_everything(params['training']['random_seed'])
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(params, augmented=True)
    
    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    drug_dim = sample_batch['drug_features'].shape[1]
    target_dim = sample_batch['target_features'].shape[1]
    
    # Create model with trial suggestions
    model = OptimizableModel(trial, drug_dim, target_dim)
    
    # Suggest training hyperparameters
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    accumulate_grad_batches = trial.suggest_int("accumulate_grad_batches", 1, 4)
    
    # Update data loaders with new batch size
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=params['training']['num_workers'],
        pin_memory=True
    )
    
    # Set up MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name="drug_repurposing_hyperparameter_tuning",
        tracking_uri=mlflow.get_tracking_uri(),
        run_name=f"optuna_trial_{trial.number}"
    )
    
    # Log hyperparameters to MLflow
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(trial.params)
        mlflow.log_param("trial_number", trial.number)
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max'
    )
    
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_auc")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(f"models/optuna/trial_{trial.number}"),
        filename="checkpoint-{epoch:02d}-{val_auc:.3f}",
        monitor='val_auc',
        mode='max',
        save_top_k=1
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=30,  # Shorter for hyperparameter search
        gpus=1 if torch.cuda.is_available() else 0,
        logger=mlflow_logger,
        callbacks=[early_stopping, pruning_callback, checkpoint_callback],
        gradient_clip_val=params['training']['gradient_clip_val'],
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=False,  # Disable for cleaner output
        deterministic=True
    )
    
    try:
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Get best validation AUC
        best_val_auc = trainer.callback_metrics.get("val_auc", 0.0)
        
        # Log final metrics to MLflow
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_metric("best_val_auc", best_val_auc)
            mlflow.log_metric("final_epoch", trainer.current_epoch)
            
        return best_val_auc
        
    except optuna.TrialPruned:
        # Log pruning to MLflow
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_param("pruned", True)
            mlflow.log_metric("pruned_epoch", trainer.current_epoch)
        raise

def run_hyperparameter_optimization(n_trials=50):
    """Run Optuna hyperparameter optimization."""
    logger.info("Starting hyperparameter optimization...")
    
    # Load parameters
    params = load_params()
    
    # Set up MLflow
    mlflow.set_tracking_uri(f"file://{Path.cwd()}/mlruns")
    mlflow.set_experiment("drug_repurposing_hyperparameter_tuning")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=params['training']['random_seed']),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Add default hyperparameters from params.yaml
    study.enqueue_trial({
        "hidden_dim": params['model']['hidden_dim'],
        "dropout": params['model']['dropout'],
        "lr": params['training']['learning_rate'],
        "batch_size": params['training']['batch_size'],
        "accumulate_grad_batches": params['training']['accumulate_grad_batches'],
        "n_layers": 3,
        "activation": "ReLU"
    })
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, params),
        n_trials=n_trials,
        timeout=None,
        catch=(Exception,),
        callbacks=[lambda study, trial: logger.info(f"Trial {trial.number} finished with value: {trial.value}")]
    )
    
    # Log study results to MLflow
    with mlflow.start_run(run_name="hyperparameter_study_summary"):
        # Log best trial
        best_trial = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_val_auc", best_trial.value)
        mlflow.log_metric("best_trial_number", best_trial.number)
        
        # Log study statistics
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_complete_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
        mlflow.log_metric("n_pruned_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
        
        # Save study results
        study_results = {
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "n_trials": len(study.trials),
            "study_name": study.study_name,
            "datetime": datetime.now().isoformat()
        }
        
        results_file = Path("models/optuna/study_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(study_results, f, indent=2)
            
        mlflow.log_artifact(results_file)
        
        # Create and log importance plot
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history
            
            # Parameter importance
            fig1 = plot_param_importances(study)
            fig1.savefig("models/optuna/param_importances.png")
            mlflow.log_artifact("models/optuna/param_importances.png")
            
            # Optimization history
            fig2 = plot_optimization_history(study)
            fig2.savefig("models/optuna/optimization_history.png")
            mlflow.log_artifact("models/optuna/optimization_history.png")
            
            plt.close('all')
        except:
            logger.warning("Could not create optimization plots")
    
    return study, best_trial

def main():
    """Main function for hyperparameter optimization."""
    logger.info("🚀 Starting Hyperparameter Optimization with Optuna")
    
    try:
        # Check for command line arguments
        n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        
        # Run optimization
        study, best_trial = run_hyperparameter_optimization(n_trials=n_trials)
        
        print("\n✅ Hyperparameter Optimization Complete!")
        print("=" * 60)
        print(f"🏆 Best trial: {best_trial.number}")
        print(f"📊 Best validation AUC: {best_trial.value:.4f}")
        print("\n📋 Best hyperparameters:")
        for param, value in best_trial.params.items():
            print(f"  - {param}: {value}")
        print("\n💾 Results saved to: models/optuna/study_results.json")
        print("📊 Access MLflow UI at: http://127.0.0.1:5000")
        print("🔍 View experiment: drug_repurposing_hyperparameter_tuning")
        
        return 0
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())