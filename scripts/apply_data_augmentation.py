#!/usr/bin/env python3
"""
Data Augmentation Techniques for Drug-Target Interaction Data

This script applies various data augmentation techniques to increase the diversity
and size of the training dataset for improved model performance.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
import yaml
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import random

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# RDKit imports for chemical augmentation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available, chemical augmentation will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

class DataAugmentationEngine:
    """Engine for applying various data augmentation techniques."""
    
    def __init__(self, params):
        self.params = params
        self.random_state = params['preprocess']['random_state']
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
    def smote_augmentation(self, features, labels, target_class=None, n_samples=None):
        """Apply SMOTE (Synthetic Minority Oversampling Technique) augmentation."""
        logger.info("Applying SMOTE augmentation...")
        
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning("imbalanced-learn not available, skipping SMOTE")
            return features, labels
        
        if target_class is not None:
            # Focus on specific class
            mask = (labels == target_class)
            if mask.sum() < 2:
                logger.warning(f"Not enough samples for class {target_class}, skipping SMOTE")
                return features, labels
            
            class_features = features[mask]
            class_labels = labels[mask]
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, len(class_features)-1))
            try:
                augmented_features, augmented_labels = smote.fit_resample(class_features, class_labels)
                return augmented_features, augmented_labels
            except ValueError as e:
                logger.warning(f"SMOTE failed for class {target_class}: {e}")
                return features, labels
        else:
            # Apply to all classes
            smote = SMOTE(random_state=self.random_state)
            try:
                augmented_features, augmented_labels = smote.fit_resample(features, labels)
                return augmented_features, augmented_labels
            except ValueError as e:
                logger.warning(f"SMOTE failed: {e}")
                return features, labels
    
    def noise_injection(self, features, noise_factor=0.01, n_augmented=None):
        """Add Gaussian noise to features for augmentation."""
        logger.info("Applying noise injection augmentation...")
        
        if n_augmented is None:
            n_augmented = len(features)
        
        # Sample indices for augmentation
        indices = np.random.choice(len(features), size=n_augmented, replace=True)
        
        augmented_features = []
        for idx in indices:
            feature_vec = features[idx].copy()
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_factor * np.std(feature_vec), feature_vec.shape)
            augmented_vec = feature_vec + noise
            
            # Ensure binary features stay binary (for fingerprints)
            if np.all(np.isin(feature_vec, [0, 1])):
                augmented_vec = np.clip(augmented_vec, 0, 1)
                augmented_vec = (augmented_vec > 0.5).astype(int)
            
            augmented_features.append(augmented_vec)
        
        return np.array(augmented_features)
    
    def feature_dropout(self, features, dropout_rate=0.1, n_augmented=None):
        """Apply feature dropout for augmentation."""
        logger.info("Applying feature dropout augmentation...")
        
        if n_augmented is None:
            n_augmented = len(features)
        
        indices = np.random.choice(len(features), size=n_augmented, replace=True)
        
        augmented_features = []
        for idx in indices:
            feature_vec = features[idx].copy()
            
            # Randomly dropout features
            dropout_mask = np.random.random(len(feature_vec)) > dropout_rate
            augmented_vec = feature_vec * dropout_mask
            
            augmented_features.append(augmented_vec)
        
        return np.array(augmented_features)
    
    def chemical_enumeration(self, smiles_list, max_enum=5):
        """Generate chemical enumerations of SMILES strings."""
        logger.info("Applying chemical enumeration...")
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping chemical enumeration")
            return smiles_list
        
        augmented_smiles = []
        
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '' or smiles == 'N/A':
                augmented_smiles.append(smiles)
                continue
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Generate different SMILES representations
                    enum_smiles = [smiles]  # Include original
                    
                    for _ in range(max_enum - 1):
                        try:
                            # Generate randomized SMILES
                            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                            if random_smiles != smiles and random_smiles not in enum_smiles:
                                enum_smiles.append(random_smiles)
                        except:
                            continue
                    
                    augmented_smiles.extend(enum_smiles)
                else:
                    augmented_smiles.append(smiles)
            except:
                augmented_smiles.append(smiles)
        
        return augmented_smiles
    
    def bootstrap_sampling(self, data, n_bootstrap=None):
        """Apply bootstrap sampling for data augmentation."""
        logger.info("Applying bootstrap sampling...")
        
        if n_bootstrap is None:
            n_bootstrap = len(data)
        
        # Bootstrap sampling with replacement
        bootstrapped_indices = np.random.choice(len(data), size=n_bootstrap, replace=True)
        bootstrapped_data = [data[i] for i in bootstrapped_indices]
        
        return bootstrapped_data
    
    def class_balancing(self, data, target_key='interaction_type'):
        """Balance classes by oversampling minority classes."""
        logger.info("Applying class balancing...")
        
        # Group by interaction type
        class_groups = {}
        for entry in data:
            interaction_type = entry.get(target_key, 'unknown')
            if interaction_type not in class_groups:
                class_groups[interaction_type] = []
            class_groups[interaction_type].append(entry)
        
        # Find the maximum class size
        max_size = max(len(group) for group in class_groups.values())
        
        balanced_data = []
        for interaction_type, group in class_groups.items():
            # Oversample to match max size
            if len(group) < max_size:
                oversampled = resample(group, n_samples=max_size, random_state=self.random_state, replace=True)
                balanced_data.extend(oversampled)
            else:
                balanced_data.extend(group)
        
        # Shuffle the balanced data
        np.random.shuffle(balanced_data)
        
        return balanced_data

def augment_labeled_dataset():
    """Apply augmentation to the labeled dataset."""
    logger.info("Starting data augmentation process...")
    
    # Load parameters
    params = load_params()
    
    # Initialize augmentation engine
    aug_engine = DataAugmentationEngine(params)
    
    # Load training data
    labeled_dir = Path("data/labeled")
    train_file = labeled_dir / "train_dataset.json"
    
    if not train_file.exists():
        logger.error("Training dataset not found. Please run import_labeled_datasets.py first.")
        return None
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    logger.info(f"Original training data size: {len(train_data)}")
    
    # Apply class balancing
    balanced_data = aug_engine.class_balancing(train_data)
    logger.info(f"After class balancing: {len(balanced_data)}")
    
    # Apply bootstrap sampling for additional diversity
    bootstrap_data = aug_engine.bootstrap_sampling(balanced_data, n_bootstrap=int(len(balanced_data) * 1.5))
    logger.info(f"After bootstrap sampling: {len(bootstrap_data)}")
    
    # Chemical enumeration for SMILES
    smiles_list = [entry['drug_metadata']['smiles'] for entry in bootstrap_data]
    enumerated_smiles = aug_engine.chemical_enumeration(smiles_list, max_enum=2)
    
    # Create augmented entries
    augmented_data = []
    smiles_idx = 0
    
    for entry in bootstrap_data:
        # Original entry
        augmented_data.append(entry)
        
        # Enumerate chemical representations
        original_smiles = entry['drug_metadata']['smiles']
        entry_enums = []
        
        while smiles_idx < len(enumerated_smiles) and len(entry_enums) < 2:
            if enumerated_smiles[smiles_idx] != original_smiles:
                enum_entry = entry.copy()
                enum_entry['drug_metadata'] = enum_entry['drug_metadata'].copy()
                enum_entry['drug_metadata']['smiles'] = enumerated_smiles[smiles_idx]
                enum_entry['source'] = f"{entry['source']}_enumerated"
                entry_enums.append(enum_entry)
            smiles_idx += 1
        
        augmented_data.extend(entry_enums)
    
    logger.info(f"After chemical enumeration: {len(augmented_data)}")
    
    return augmented_data

def augment_features():
    """Apply feature-level augmentation."""
    logger.info("Starting feature-level augmentation...")
    
    # Load parameters
    params = load_params()
    aug_engine = DataAugmentationEngine(params)
    
    # Load features
    features_dir = Path("data/processed/features")
    train_features_dir = features_dir / "train"
    
    if not train_features_dir.exists():
        logger.error("Training features not found. Please run generate_molecular_fingerprints.py first.")
        return None
    
    augmented_features = {}
    
    # Load and augment each feature type
    for feature_file in train_features_dir.glob("*_features.npy"):
        feature_type = feature_file.stem.replace('_features', '')
        features = np.load(feature_file)
        
        logger.info(f"Augmenting {feature_type} features: {features.shape}")
        
        # Apply different augmentation techniques
        if 'morgan' in feature_type or 'fingerprint' in feature_type:
            # For binary fingerprints, use feature dropout
            augmented = aug_engine.feature_dropout(features, dropout_rate=0.05, n_augmented=len(features))
        else:
            # For continuous features, use noise injection
            augmented = aug_engine.noise_injection(features, noise_factor=0.01, n_augmented=len(features))
        
        # Combine original and augmented
        combined_features = np.vstack([features, augmented])
        augmented_features[feature_type] = combined_features
        
        logger.info(f"Augmented {feature_type} features: {combined_features.shape}")
    
    # Save augmented features
    aug_dir = features_dir / "train_augmented"
    aug_dir.mkdir(exist_ok=True)
    
    for feature_type, features in augmented_features.items():
        np.save(aug_dir / f"{feature_type}_features.npy", features)
    
    logger.info(f"Augmented features saved to: {aug_dir}")
    
    return augmented_features

def create_augmentation_summary(original_data, augmented_data, augmented_features):
    """Create a summary of augmentation results."""
    logger.info("Creating augmentation summary...")
    
    summary = {
        "augmentation_timestamp": datetime.now().isoformat(),
        "original_data_size": len(original_data) if original_data else 0,
        "augmented_data_size": len(augmented_data) if augmented_data else 0,
        "augmentation_ratio": len(augmented_data) / len(original_data) if original_data and augmented_data else 0,
        "techniques_applied": [
            "class_balancing",
            "bootstrap_sampling", 
            "chemical_enumeration",
            "feature_noise_injection",
            "feature_dropout"
        ],
        "feature_augmentation": {}
    }
    
    if augmented_features:
        for feature_type, features in augmented_features.items():
            summary["feature_augmentation"][feature_type] = {
                "original_shape": f"({features.shape[0]//2}, {features.shape[1]})",
                "augmented_shape": str(features.shape),
                "augmentation_factor": 2.0
            }
    
    # Class distribution analysis
    if augmented_data:
        class_dist = {}
        for entry in augmented_data:
            interaction_type = entry.get('interaction_type', 'unknown')
            class_dist[interaction_type] = class_dist.get(interaction_type, 0) + 1
        
        summary["class_distribution"] = class_dist
    
    # Save summary
    output_dir = Path("data/processed")
    summary_file = output_dir / "augmentation_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Augmentation summary saved to: {summary_file}")
    
    return summary

def main():
    """Main augmentation function."""
    logger.info("Starting data augmentation pipeline...")
    
    try:
        # Load original training data
        labeled_dir = Path("data/labeled")
        train_file = labeled_dir / "train_dataset.json"
        
        original_data = []
        if train_file.exists():
            with open(train_file, 'r') as f:
                original_data = json.load(f)
        
        # Apply data-level augmentation
        augmented_data = augment_labeled_dataset()
        
        # Save augmented dataset
        if augmented_data:
            aug_file = labeled_dir / "train_augmented_dataset.json"
            with open(aug_file, 'w') as f:
                json.dump(augmented_data, f, indent=2)
            logger.info(f"Augmented dataset saved to: {aug_file}")
        
        # Apply feature-level augmentation
        augmented_features = augment_features()
        
        # Create summary
        summary = create_augmentation_summary(original_data, augmented_data, augmented_features)
        
        # Print results
        print("\n🔄 Data Augmentation Complete!")
        print("=" * 50)
        print(f"📊 Original Training Size: {summary['original_data_size']}")
        print(f"📈 Augmented Training Size: {summary['augmented_data_size']}")
        print(f"📊 Augmentation Ratio: {summary['augmentation_ratio']:.2f}x")
        print()
        print("🧪 Techniques Applied:")
        for technique in summary['techniques_applied']:
            print(f"  ✅ {technique.replace('_', ' ').title()}")
        print()
        print("🧬 Feature Augmentation:")
        for feature_type, info in summary.get('feature_augmentation', {}).items():
            print(f"  {feature_type}: {info['original_shape']} → {info['augmented_shape']}")
        print()
        print("📊 Class Distribution:")
        for cls, count in summary.get('class_distribution', {}).items():
            print(f"  {cls}: {count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data augmentation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())