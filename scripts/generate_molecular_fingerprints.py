#!/usr/bin/env python3
"""
Molecular Fingerprint Generation for Drug-Target Interaction Prediction

This script generates various types of molecular fingerprints and target features
for use in machine learning models.
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

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

class MolecularFingerprintGenerator:
    """Generate various molecular fingerprints from SMILES."""
    
    def __init__(self, params):
        self.params = params
        self.fingerprint_size = params['features']['fingerprint_size']
        self.radius = params['features']['radius']
        self.use_features = params['features']['use_features']
        
    def smiles_to_mol(self, smiles):
        """Convert SMILES string to RDKit molecule object."""
        try:
            if pd.isna(smiles) or smiles == '' or smiles == 'N/A':
                return None
            mol = Chem.MolFromSmiles(str(smiles))
            return mol
        except:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
    
    def generate_morgan_fingerprints(self, smiles_list):
        """Generate Morgan (ECFP) fingerprints."""
        logger.info("Generating Morgan fingerprints...")
        
        fingerprints = []
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                # Generate Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.fingerprint_size
                )
                # Convert to numpy array
                fp_array = np.zeros((self.fingerprint_size,))
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                fingerprints.append(fp_array)
            else:
                # Return zero vector for invalid molecules
                fingerprints.append(np.zeros(self.fingerprint_size))
        
        return np.array(fingerprints)
    
    def generate_rdkit_fingerprints(self, smiles_list):
        """Generate RDKit fingerprints."""
        logger.info("Generating RDKit fingerprints...")
        
        fingerprints = []
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                # Generate RDKit fingerprint
                fp = FingerprintMols.FingerprintMol(mol)
                # Convert to bit vector with fixed size
                fp_bits = fp.ToBitString()
                # Pad or truncate to desired size
                if len(fp_bits) > self.fingerprint_size:
                    fp_bits = fp_bits[:self.fingerprint_size]
                else:
                    fp_bits = fp_bits.ljust(self.fingerprint_size, '0')
                
                fp_array = np.array([int(bit) for bit in fp_bits])
                fingerprints.append(fp_array)
            else:
                fingerprints.append(np.zeros(self.fingerprint_size))
        
        return np.array(fingerprints)
    
    def generate_molecular_descriptors(self, smiles_list):
        """Generate molecular descriptors."""
        logger.info("Generating molecular descriptors...")
        
        descriptors = []
        descriptor_names = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
            'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'FractionCsp3',
            'HeavyAtomCount', 'NumHeteroatoms', 'TPSA', 'LabuteASA', 'BalabanJ',
            'BertzCT', 'Chi0', 'Chi1', 'HallKierAlpha', 'Ipc', 'Kappa1',
            'Kappa2', 'Kappa3', 'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
            'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3'
        ]
        
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                try:
                    desc_values = []
                    
                    # Basic descriptors
                    desc_values.append(Descriptors.MolWt(mol))
                    desc_values.append(Crippen.MolLogP(mol))
                    desc_values.append(Descriptors.NumHDonors(mol))
                    desc_values.append(Descriptors.NumHAcceptors(mol))
                    desc_values.append(Descriptors.NumRotatableBonds(mol))
                    desc_values.append(Descriptors.NumAromaticRings(mol))
                    desc_values.append(Descriptors.NumAliphaticRings(mol))
                    desc_values.append(Descriptors.RingCount(mol))
                    desc_values.append(Descriptors.FractionCsp3(mol))
                    desc_values.append(Descriptors.HeavyAtomCount(mol))
                    desc_values.append(Descriptors.NumHeteroatoms(mol))
                    desc_values.append(Descriptors.TPSA(mol))
                    desc_values.append(Descriptors.LabuteASA(mol))
                    desc_values.append(Descriptors.BalabanJ(mol))
                    desc_values.append(Descriptors.BertzCT(mol))
                    desc_values.append(Descriptors.Chi0(mol))
                    desc_values.append(Descriptors.Chi1(mol))
                    desc_values.append(Descriptors.HallKierAlpha(mol))
                    desc_values.append(Descriptors.Ipc(mol))
                    desc_values.append(Descriptors.Kappa1(mol))
                    desc_values.append(Descriptors.Kappa2(mol))
                    desc_values.append(Descriptors.Kappa3(mol))
                    desc_values.append(Descriptors.MaxEStateIndex(mol))
                    desc_values.append(Descriptors.MinEStateIndex(mol))
                    desc_values.append(Descriptors.MaxAbsEStateIndex(mol))
                    desc_values.append(Descriptors.MaxPartialCharge(mol))
                    desc_values.append(Descriptors.MinPartialCharge(mol))
                    desc_values.append(Descriptors.MaxAbsPartialCharge(mol))
                    desc_values.append(Descriptors.FpDensityMorgan1(mol))
                    desc_values.append(Descriptors.FpDensityMorgan2(mol))
                    desc_values.append(Descriptors.FpDensityMorgan3(mol))
                    
                    # Handle NaN values
                    desc_values = [0.0 if pd.isna(val) or np.isinf(val) else float(val) for val in desc_values]
                    descriptors.append(desc_values)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate descriptors for SMILES {smiles}: {e}")
                    descriptors.append([0.0] * len(descriptor_names))
            else:
                descriptors.append([0.0] * len(descriptor_names))
        
        return np.array(descriptors), descriptor_names

class TargetFeatureGenerator:
    """Generate target protein features."""
    
    def __init__(self, params):
        self.params = params
        
    def generate_target_features(self, target_data):
        """Generate features for target proteins."""
        logger.info("Generating target features...")
        
        features = []
        feature_names = []
        
        # Encode target class
        if 'target_class' in target_data.columns:
            le_class = LabelEncoder()
            target_classes = target_data['target_class'].fillna('unknown')
            class_encoded = le_class.fit_transform(target_classes)
            features.append(class_encoded.reshape(-1, 1))
            feature_names.append('target_class_encoded')
            
            # One-hot encode target class
            class_unique = le_class.classes_
            for i, class_name in enumerate(class_unique):
                class_binary = (class_encoded == i).astype(int)
                features.append(class_binary.reshape(-1, 1))
                feature_names.append(f'target_class_{class_name}')
        
        # Encode organism
        if 'organism' in target_data.columns:
            le_organism = LabelEncoder()
            organisms = target_data['organism'].fillna('unknown')
            organism_encoded = le_organism.fit_transform(organisms)
            features.append(organism_encoded.reshape(-1, 1))
            feature_names.append('organism_encoded')
        
        # UniProt ID length (proxy for protein complexity)
        if 'uniprot_id' in target_data.columns:
            uniprot_lengths = target_data['uniprot_id'].fillna('').str.len()
            features.append(uniprot_lengths.values.reshape(-1, 1))
            feature_names.append('uniprot_id_length')
        
        # Target name length and word count (proxy for annotation quality)
        if 'target_name' in target_data.columns:
            name_lengths = target_data['target_name'].fillna('').str.len()
            word_counts = target_data['target_name'].fillna('').str.split().str.len()
            features.append(name_lengths.values.reshape(-1, 1))
            features.append(word_counts.values.reshape(-1, 1))
            feature_names.extend(['target_name_length', 'target_name_word_count'])
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
        else:
            combined_features = np.zeros((len(target_data), 1))
            feature_names = ['dummy_feature']
        
        return combined_features, feature_names

def generate_interaction_features(drug_features, target_features):
    """Generate interaction features by combining drug and target features."""
    logger.info("Generating interaction features...")
    
    # Ensure same number of samples
    n_samples = min(drug_features.shape[0], target_features.shape[0])
    drug_features = drug_features[:n_samples]
    target_features = target_features[:n_samples]
    
    # Concatenate drug and target features
    interaction_features = np.hstack([drug_features, target_features])
    
    # Add interaction-specific features
    # Element-wise product (interaction between drug and target features)
    if drug_features.shape[1] == target_features.shape[1]:
        product_features = drug_features * target_features
        interaction_features = np.hstack([interaction_features, product_features])
    
    # Add statistical features
    feature_stats = []
    for i in range(n_samples):
        drug_vec = drug_features[i]
        target_vec = target_features[i]
        
        # Basic statistics
        stats = [
            np.mean(drug_vec),
            np.std(drug_vec),
            np.mean(target_vec),
            np.std(target_vec),
            np.dot(drug_vec, target_vec) if len(drug_vec) == len(target_vec) else 0.0,
            np.linalg.norm(drug_vec) if len(drug_vec) > 0 else 0.0,
            np.linalg.norm(target_vec) if len(target_vec) > 0 else 0.0
        ]
        feature_stats.append(stats)
    
    feature_stats = np.array(feature_stats)
    interaction_features = np.hstack([interaction_features, feature_stats])
    
    return interaction_features

def save_features(features_dict, feature_names_dict, splits, output_dir):
    """Save generated features to files."""
    logger.info("Saving features...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features for each split
    for split_name, split_data in splits.items():
        if not split_data:
            continue
            
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Get indices for this split
        indices = list(range(len(split_data)))
        
        # Save each feature type
        for feature_type, features in features_dict.items():
            if features is not None and len(features) > 0:
                split_features = features[indices] if len(features) > len(indices) else features
                
                # Save as numpy array
                np.save(split_dir / f"{feature_type}_features.npy", split_features)
                
                # Save as CSV for inspection
                if feature_type in feature_names_dict:
                    feature_names = feature_names_dict[feature_type]
                    if len(feature_names) == split_features.shape[1]:
                        df = pd.DataFrame(split_features, columns=feature_names)
                        df.to_csv(split_dir / f"{feature_type}_features.csv", index=False)
                    else:
                        # Generic column names
                        df = pd.DataFrame(split_features)
                        df.to_csv(split_dir / f"{feature_type}_features.csv", index=False)
    
    # Save feature metadata
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "feature_types": list(features_dict.keys()),
        "feature_shapes": {k: v.shape if v is not None else None for k, v in features_dict.items()},
        "feature_names": feature_names_dict,
        "parameters": load_params()['features']
    }
    
    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save scalers
    scalers = {}
    for feature_type, features in features_dict.items():
        if features is not None and features.shape[1] > 1:
            scaler = StandardScaler()
            scaler.fit(features)
            scalers[feature_type] = scaler
    
    with open(output_dir / "scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    
    logger.info(f"Features saved to: {output_dir}")
    
    return metadata

def main():
    """Main feature generation function."""
    logger.info("Starting molecular fingerprint and feature generation...")
    
    try:
        # Load parameters
        params = load_params()
        
        # Load labeled datasets
        labeled_dir = Path("data/labeled")
        
        # Load splits
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            split_file = labeled_dir / f"{split_name}_dataset.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    splits[split_name] = json.load(f)
            else:
                splits[split_name] = []
        
        # Combine all data for feature generation
        all_data = []
        for split_data in splits.values():
            all_data.extend(split_data)
        
        if not all_data:
            logger.error("No labeled data found. Please run import_labeled_datasets.py first.")
            return 1
        
        # Extract data for feature generation
        smiles_list = [entry['drug_metadata']['smiles'] for entry in all_data]
        
        # Create target data DataFrame
        target_data = pd.DataFrame([{
            'target_class': entry['target_metadata']['target_class'],
            'organism': entry['target_metadata']['organism'],
            'uniprot_id': entry['target_metadata']['uniprot_id'],
            'target_name': entry['target_metadata']['target_name']
        } for entry in all_data])
        
        # Initialize generators
        mol_generator = MolecularFingerprintGenerator(params)
        target_generator = TargetFeatureGenerator(params)
        
        # Generate molecular features
        features_dict = {}
        feature_names_dict = {}
        
        if 'morgan_fingerprints' in params['features']['use_features']:
            morgan_features = mol_generator.generate_morgan_fingerprints(smiles_list)
            features_dict['morgan'] = morgan_features
            feature_names_dict['morgan'] = [f'morgan_{i}' for i in range(morgan_features.shape[1])]
        
        if 'molecular_descriptors' in params['features']['use_features']:
            desc_features, desc_names = mol_generator.generate_molecular_descriptors(smiles_list)
            features_dict['descriptors'] = desc_features
            feature_names_dict['descriptors'] = desc_names
        
        if 'target_features' in params['features']['use_features']:
            target_features, target_names = target_generator.generate_target_features(target_data)
            features_dict['target'] = target_features
            feature_names_dict['target'] = target_names
        
        # Generate interaction features
        if 'morgan' in features_dict and 'target' in features_dict:
            interaction_features = generate_interaction_features(
                features_dict['morgan'], 
                features_dict['target']
            )
            features_dict['interaction'] = interaction_features
            feature_names_dict['interaction'] = (
                [f'drug_{name}' for name in feature_names_dict['morgan']] +
                [f'target_{name}' for name in feature_names_dict['target']] +
                [f'product_{i}' for i in range(min(features_dict['morgan'].shape[1], features_dict['target'].shape[1]))] +
                ['drug_mean', 'drug_std', 'target_mean', 'target_std', 'dot_product', 'drug_norm', 'target_norm']
            )
        
        # Save features
        output_dir = Path("data/processed/features")
        metadata = save_features(features_dict, feature_names_dict, splits, output_dir)
        
        # Print summary
        print("\n🧬 Molecular Fingerprint Generation Complete!")
        print("=" * 60)
        print(f"📊 Total Samples Processed: {len(all_data)}")
        print(f"🧪 Feature Types Generated: {list(features_dict.keys())}")
        print()
        
        for feature_type, features in features_dict.items():
            if features is not None:
                print(f"  {feature_type.title()} Features: {features.shape}")
        
        print()
        print("📁 Output Directory: ", output_dir)
        print("🔧 Scalers and Metadata: Saved for reproducibility")
        print()
        print("📊 Data Splits Processed:")
        for split_name, split_data in splits.items():
            print(f"  {split_name.title()}: {len(split_data)} samples")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())