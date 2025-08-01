#!/usr/bin/env python3
"""
Model Inference Script for MLOps Drug Repurposing Project

This script provides utilities for loading models and running inference
on drug-target interaction data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DrugBANModel(nn.Module):
    """DrugBAN model implementation for inference."""
    
    def __init__(self, drug_dim=256, target_dim=256, hidden_dim=512, num_heads=8):
        super(DrugBANModel, self).__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Drug encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Bilinear Attention Network
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, drug_features, target_features):
        # Encode drug and target features
        drug_encoded = self.drug_encoder(drug_features)
        target_encoded = self.target_encoder(target_features)
        
        # Apply attention mechanism
        drug_attended, _ = self.attention(
            drug_encoded.unsqueeze(1),
            target_encoded.unsqueeze(1),
            target_encoded.unsqueeze(1)
        )
        target_attended, _ = self.attention(
            target_encoded.unsqueeze(1),
            drug_encoded.unsqueeze(1),
            drug_encoded.unsqueeze(1)
        )
        
        # Concatenate attended features
        combined_features = torch.cat([
            drug_attended.squeeze(1),
            target_attended.squeeze(1)
        ], dim=1)
        
        # Predict interaction probability
        output = self.classifier(combined_features)
        return output

class DrugFeatureExtractor:
    """Extract molecular features from drug SMILES."""
    
    def __init__(self, fingerprint_size=2048, radius=2):
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def smiles_to_morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """Convert SMILES to Morgan fingerprint."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return np.zeros(self.fingerprint_size)
            
            # Generate Morgan fingerprint
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.fingerprint_size
            )
            
            # Convert to numpy array
            return np.array(fingerprint)
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return np.zeros(self.fingerprint_size)
    
    def extract_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract features for a list of SMILES."""
        features = []
        for smiles in smiles_list:
            fp = self.smiles_to_morgan_fingerprint(smiles)
            features.append(fp)
        
        features = np.array(features)
        
        # Apply scaling if fitted
        if self.is_fitted:
            features = self.scaler.transform(features)
        
        return features
    
    def fit_scaler(self, smiles_list: List[str]):
        """Fit the scaler on training data."""
        features = []
        for smiles in smiles_list:
            fp = self.smiles_to_morgan_fingerprint(smiles)
            features.append(fp)
        
        features = np.array(features)
        self.scaler.fit(features)
        self.is_fitted = True
        logger.info("✅ Feature scaler fitted")

class TargetFeatureExtractor:
    """Extract features from target proteins."""
    
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        # Placeholder for protein sequence encoder
        self.protein_embeddings = {}
    
    def extract_features(self, target_ids: List[str]) -> np.ndarray:
        """Extract features for target proteins."""
        features = []
        
        for target_id in target_ids:
            if target_id in self.protein_embeddings:
                embedding = self.protein_embeddings[target_id]
            else:
                # Generate dummy embedding for demonstration
                np.random.seed(hash(target_id) % 2**32)
                embedding = np.random.normal(0, 1, self.embedding_dim)
                self.protein_embeddings[target_id] = embedding
            
            features.append(embedding)
        
        features = np.array(features)
        
        # Apply scaling if fitted
        if self.is_fitted:
            features = self.scaler.transform(features)
        
        return features
    
    def fit_scaler(self, target_ids: List[str]):
        """Fit the scaler on training data."""
        features = self.extract_features(target_ids)
        self.scaler.fit(features)
        self.is_fitted = True
        logger.info("✅ Target feature scaler fitted")

class ModelLoader:
    """Load and manage trained models."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        
    def load_model_from_path(self, model_path: str, model_name: str = None) -> torch.nn.Module:
        """Load model from local path."""
        if model_name is None:
            model_name = Path(model_path).stem
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model configuration
            model_config = checkpoint.get('model_config', {})
            
            # Create model instance
            model = DrugBANModel(
                drug_dim=model_config.get('drug_dim', 2048),
                target_dim=model_config.get('target_dim', 1024),
                hidden_dim=model_config.get('hidden_dim', 512),
                num_heads=model_config.get('num_heads', 8)
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Store model and config
            self.models[model_name] = model
            self.model_configs[model_name] = checkpoint.get('metadata', {})
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def load_model_from_mlflow(self, model_name: str, version: str = "latest") -> torch.nn.Module:
        """Load model from MLflow registry."""
        logger.info(f"Loading model {model_name} (version: {version}) from MLflow...")
        
        try:
            # Load model from MLflow
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pytorch.load_model(model_uri)
            
            # Store in cache
            self.models[model_name] = model
            
            logger.info(f"✅ Model {model_name} loaded from MLflow")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from MLflow: {e}")
            raise
    
    def get_model(self, model_name: str) -> torch.nn.Module:
        """Get loaded model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Load it first.")
        return self.models[model_name]

class DrugTargetPredictor:
    """Main class for drug-target interaction prediction."""
    
    def __init__(self, model_path: str = None, model_name: str = "DrugBAN"):
        self.model_name = model_name
        self.model_loader = ModelLoader()
        self.drug_extractor = DrugFeatureExtractor()
        self.target_extractor = TargetFeatureExtractor()
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model for inference."""
        self.model = self.model_loader.load_model_from_path(model_path, self.model_name)
        logger.info(f"✅ {self.model_name} model ready for inference")
    
    def load_model_from_mlflow(self, model_name: str, version: str = "latest"):
        """Load model from MLflow."""
        self.model = self.model_loader.load_model_from_mlflow(model_name, version)
        self.model_name = model_name
        logger.info(f"✅ {model_name} model ready for inference")
    
    def predict_single(self, drug_smiles: str, target_id: str) -> float:
        """Predict interaction probability for a single drug-target pair."""
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Extract features
        drug_features = self.drug_extractor.extract_features([drug_smiles])
        target_features = self.target_extractor.extract_features([target_id])
        
        # Convert to tensors
        drug_tensor = torch.FloatTensor(drug_features)
        target_tensor = torch.FloatTensor(target_features)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(drug_tensor, target_tensor)
            probability = prediction.item()
        
        return probability
    
    def predict_batch(self, drug_smiles_list: List[str], target_ids_list: List[str]) -> np.ndarray:
        """Predict interaction probabilities for multiple drug-target pairs."""
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        if len(drug_smiles_list) != len(target_ids_list):
            raise ValueError("Drug and target lists must have the same length")
        
        # Extract features
        drug_features = self.drug_extractor.extract_features(drug_smiles_list)
        target_features = self.target_extractor.extract_features(target_ids_list)
        
        # Convert to tensors
        drug_tensor = torch.FloatTensor(drug_features)
        target_tensor = torch.FloatTensor(target_features)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(drug_tensor, target_tensor)
            probabilities = predictions.numpy().flatten()
        
        return probabilities
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                              drug_col: str = "smiles", 
                              target_col: str = "target_id") -> pd.DataFrame:
        """Predict interactions from a dataframe."""
        logger.info(f"Running predictions on {len(df)} drug-target pairs...")
        
        # Extract drug and target data
        drug_smiles = df[drug_col].tolist()
        target_ids = df[target_col].tolist()
        
        # Run predictions
        probabilities = self.predict_batch(drug_smiles, target_ids)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['interaction_probability'] = probabilities
        result_df['predicted_interaction'] = (probabilities > 0.5).astype(int)
        
        logger.info(f"✅ Predictions completed")
        return result_df

def load_test_data() -> pd.DataFrame:
    """Load test data for demonstration."""
    # Use sample data from our project
    sample_file = Path("data/raw/sample/drug_target_interactions.csv")
    
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        
        # Load drug metadata to get SMILES
        drug_file = Path("data/raw/sample/drug_metadata.csv")
        if drug_file.exists():
            drug_df = pd.read_csv(drug_file)
            df = df.merge(drug_df[['drug_id', 'smiles']], on='drug_id', how='left')
        
        return df
    else:
        # Create dummy test data
        test_data = {
            'drug_id': ['CHEMBL25', 'CHEMBL53', 'CHEMBL85'],
            'target_id': ['ENSP001', 'ENSP002', 'ENSP003'],
            'smiles': [
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
                'CC(C)CC1=CC=C(C=C1)C(C(=O)O)C',  # Ibuprofen
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
            ],
            'interaction_type': ['inhibitor', 'agonist', 'antagonist']
        }
        
        return pd.DataFrame(test_data)

def main():
    """Main function for testing model inference."""
    logger.info("🚀 Testing model inference capabilities...")
    
    try:
        # Initialize predictor
        model_path = "models/pretrained/drugban_baseline.pth"
        predictor = DrugTargetPredictor(model_path)
        
        # Load test data
        test_df = load_test_data()
        logger.info(f"📊 Loaded {len(test_df)} test samples")
        
        # Test single prediction
        if len(test_df) > 0:
            row = test_df.iloc[0]
            drug_smiles = row.get('smiles', 'CC(=O)OC1=CC=CC=C1C(=O)O')
            target_id = row['target_id']
            
            logger.info(f"🧪 Testing single prediction...")
            probability = predictor.predict_single(drug_smiles, target_id)
            logger.info(f"  Drug: {row.get('drug_id', 'unknown')} ({drug_smiles})")
            logger.info(f"  Target: {target_id}")
            logger.info(f"  Interaction probability: {probability:.4f}")
        
        # Test batch prediction
        if 'smiles' in test_df.columns:
            logger.info(f"🎯 Testing batch prediction...")
            results_df = predictor.predict_from_dataframe(test_df)
            
            # Display results
            logger.info("📈 Prediction results:")
            for idx, row in results_df.iterrows():
                logger.info(f"  {row['drug_id']} -> {row['target_id']}: {row['interaction_probability']:.4f}")
            
            # Save results
            results_path = Path("models/baseline_results/inference_test_results.csv")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path, index=False)
            logger.info(f"💾 Results saved to: {results_path}")
        
        logger.info("✅ Model inference testing completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model inference testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())