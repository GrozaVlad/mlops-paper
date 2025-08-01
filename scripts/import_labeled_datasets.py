#!/usr/bin/env python3
"""
Import Labeled Datasets for Drug-Target Interaction Annotation

This script imports existing datasets with labels into the Label Studio format
and creates an organized dataset structure for the MLOps pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def import_biosnap_data():
    """Import and process BIOSNAP dataset with labels."""
    logger.info("Importing BIOSNAP dataset...")
    
    biosnap_dir = Path("data/raw/biosnap")
    labeled_data = []
    
    try:
        # Load BIOSNAP drug-target interactions
        interactions_file = biosnap_dir / "drug_target_interactions.csv"
        drug_metadata_file = biosnap_dir / "drug_metadata.csv"
        target_metadata_file = biosnap_dir / "target_metadata.csv"
        
        if interactions_file.exists():
            df_interactions = pd.read_csv(interactions_file)
            df_drugs = pd.read_csv(drug_metadata_file) if drug_metadata_file.exists() else pd.DataFrame()
            df_targets = pd.read_csv(target_metadata_file) if target_metadata_file.exists() else pd.DataFrame()
            
            # Merge datasets
            if not df_drugs.empty:
                df_interactions = df_interactions.merge(df_drugs, on='drug_id', how='left')
            if not df_targets.empty:
                df_interactions = df_interactions.merge(df_targets, on='target_id', how='left')
            
            # Convert to labeled format
            for idx, row in df_interactions.iterrows():
                labeled_entry = {
                    "source": "biosnap",
                    "drug_id": row['drug_id'],
                    "target_id": row['target_id'],
                    "interaction_type": row.get('interaction_type', 'binding'),
                    "binding_affinity": row.get('binding_affinity', None),
                    "confidence_score": row.get('confidence_score', 0.5),
                    "evidence_quality": "medium",  # BIOSNAP is generally medium quality
                    "annotation_status": "pre_labeled",
                    "drug_metadata": {
                        "drug_name": row.get('drug_name', ''),
                        "smiles": row.get('smiles', ''),
                        "molecular_weight": row.get('molecular_weight', None),
                        "logp": row.get('logp', None)
                    },
                    "target_metadata": {
                        "target_name": row.get('target_name', ''),
                        "uniprot_id": row.get('uniprot_id', ''),
                        "organism": row.get('organism', 'Homo sapiens'),
                        "target_class": row.get('target_class', '')
                    }
                }
                labeled_data.append(labeled_entry)
            
            logger.info(f"Imported {len(labeled_data)} interactions from BIOSNAP")
            
    except Exception as e:
        logger.warning(f"Could not import BIOSNAP data: {e}")
    
    return labeled_data

def import_bindingdb_data():
    """Import and process BindingDB dataset with labels."""
    logger.info("Importing BindingDB dataset...")
    
    bindingdb_dir = Path("data/raw/bindingdb")
    labeled_data = []
    
    try:
        # Load BindingDB data
        binding_file = bindingdb_dir / "binding_data.tsv"
        compounds_file = bindingdb_dir / "compounds.tsv"
        targets_file = bindingdb_dir / "targets.tsv"
        
        if binding_file.exists():
            df_binding = pd.read_csv(binding_file, sep='\t')
            df_compounds = pd.read_csv(compounds_file, sep='\t') if compounds_file.exists() else pd.DataFrame()
            df_targets = pd.read_csv(targets_file, sep='\t') if targets_file.exists() else pd.DataFrame()
            
            # Sample a subset for demonstration (BindingDB is very large)
            if len(df_binding) > 100:
                df_binding = df_binding.sample(n=100, random_state=42)
            
            # Process binding data
            for idx, row in df_binding.iterrows():
                # Determine interaction type based on binding data
                interaction_type = "binding"
                if 'Ki' in df_binding.columns and pd.notna(row.get('Ki')):
                    interaction_type = "inhibition"
                elif 'IC50' in df_binding.columns and pd.notna(row.get('IC50')):
                    interaction_type = "inhibition"
                elif 'EC50' in df_binding.columns and pd.notna(row.get('EC50')):
                    interaction_type = "activation"
                
                # Calculate binding affinity (convert to pKd/pIC50 if available)
                binding_affinity = None
                if 'Ki' in df_binding.columns and pd.notna(row.get('Ki')):
                    try:
                        ki_value = float(row['Ki'])
                        binding_affinity = -np.log10(ki_value * 1e-9)  # Convert to pKi
                    except:
                        pass
                
                labeled_entry = {
                    "source": "bindingdb",
                    "drug_id": row.get('drug_id', f"bindingdb_drug_{idx}"),
                    "target_id": row.get('target_id', f"bindingdb_target_{idx}"),
                    "interaction_type": interaction_type,
                    "binding_affinity": binding_affinity,
                    "confidence_score": 0.8,  # BindingDB is generally high quality
                    "evidence_quality": "high",
                    "annotation_status": "pre_labeled",
                    "drug_metadata": {
                        "drug_name": row.get('compound_name', ''),
                        "smiles": row.get('smiles', ''),
                        "molecular_weight": row.get('molecular_weight', None),
                        "logp": row.get('logp', None)
                    },
                    "target_metadata": {
                        "target_name": row.get('target_name', ''),
                        "uniprot_id": row.get('uniprot_id', ''),
                        "organism": row.get('organism', 'Homo sapiens'),
                        "target_class": row.get('target_class', '')
                    }
                }
                labeled_data.append(labeled_entry)
            
            logger.info(f"Imported {len(labeled_data)} interactions from BindingDB")
            
    except Exception as e:
        logger.warning(f"Could not import BindingDB data: {e}")
    
    return labeled_data

def import_sample_data():
    """Import sample data and convert to labeled format."""
    logger.info("Importing sample dataset...")
    
    sample_dir = Path("data/raw/sample")
    labeled_data = []
    
    try:
        # Load sample data
        interactions_df = pd.read_csv(sample_dir / "drug_target_interactions.csv")
        drug_metadata_df = pd.read_csv(sample_dir / "drug_metadata.csv")
        target_metadata_df = pd.read_csv(sample_dir / "target_metadata.csv")
        
        # Merge data
        merged_df = interactions_df.merge(
            drug_metadata_df, on='drug_id', how='left'
        ).merge(
            target_metadata_df, on='target_id', how='left'
        )
        
        # Convert to labeled format
        for idx, row in merged_df.iterrows():
            labeled_entry = {
                "source": "sample",
                "drug_id": row['drug_id'],
                "target_id": row['target_id'],
                "interaction_type": row['interaction_type'],
                "binding_affinity": row.get('binding_affinity', None),
                "confidence_score": row.get('confidence_score', 0.7),
                "evidence_quality": "medium",
                "annotation_status": "verified",
                "drug_metadata": {
                    "drug_name": row['drug_name'],
                    "smiles": row['smiles'],
                    "molecular_weight": row.get('molecular_weight', None),
                    "logp": row.get('logp', None)
                },
                "target_metadata": {
                    "target_name": row['target_name'],
                    "uniprot_id": row['uniprot_id'],
                    "organism": row.get('organism', 'Homo sapiens'),
                    "target_class": row.get('target_class', '')
                }
            }
            labeled_data.append(labeled_entry)
        
        logger.info(f"Imported {len(labeled_data)} interactions from sample data")
        
    except Exception as e:
        logger.error(f"Failed to import sample data: {e}")
    
    return labeled_data

def create_train_test_splits(labeled_data, params):
    """Create train/validation/test splits from labeled data."""
    logger.info("Creating train/validation/test splits...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(labeled_data)
    
    # Get split parameters
    test_size = params['preprocess']['test_size']
    validation_size = params['preprocess'].get('validation_size', 0.2)
    random_state = params['preprocess']['random_state']
    stratify = params['preprocess'].get('stratify', True)
    
    # Stratify by interaction type if enabled and feasible
    stratify_column = None
    if stratify and 'interaction_type' in df.columns and len(df) >= 10:
        # Check if all classes have at least 2 members
        class_counts = df['interaction_type'].value_counts()
        if class_counts.min() >= 2:
            stratify_column = df['interaction_type']
        else:
            logger.warning("Disabling stratification due to insufficient samples per class")
    
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_column
    )
    
    # Second split: train vs validation
    # Adjust validation size relative to train+val set
    val_size_adjusted = validation_size / (1 - test_size)
    
    stratify_train_val = None
    if stratify and 'interaction_type' in train_val_df.columns and len(train_val_df) >= 10:
        # Check if all classes have at least 2 members in train_val set
        class_counts_train_val = train_val_df['interaction_type'].value_counts()
        if class_counts_train_val.min() >= 2:
            stratify_train_val = train_val_df['interaction_type']
        else:
            logger.warning("Disabling stratification for train/val split due to insufficient samples per class")
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_train_val
    )
    
    splits = {
        'train': train_df.to_dict('records'),
        'validation': val_df.to_dict('records'),
        'test': test_df.to_dict('records')
    }
    
    # Log split statistics
    logger.info(f"Data splits created:")
    logger.info(f"  Train: {len(splits['train'])} samples")
    logger.info(f"  Validation: {len(splits['validation'])} samples")
    logger.info(f"  Test: {len(splits['test'])} samples")
    
    # Create interaction type distribution analysis
    for split_name, split_data in splits.items():
        if split_data:
            split_df = pd.DataFrame(split_data)
            if 'interaction_type' in split_df.columns:
                type_dist = split_df['interaction_type'].value_counts()
                logger.info(f"  {split_name.capitalize()} interaction types: {type_dist.to_dict()}")
    
    return splits

def save_labeled_datasets(splits, labeled_data_all):
    """Save labeled datasets in multiple formats."""
    logger.info("Saving labeled datasets...")
    
    # Create labeled data directory
    labeled_dir = Path("data/labeled")
    labeled_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete labeled dataset
    with open(labeled_dir / "complete_labeled_dataset.json", 'w') as f:
        json.dump(labeled_data_all, f, indent=2)
    
    # Save splits
    for split_name, split_data in splits.items():
        split_file = labeled_dir / f"{split_name}_dataset.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Also save as CSV for easy inspection
        if split_data:
            split_df = pd.DataFrame(split_data)
            
            # Flatten nested dictionaries for CSV
            flat_data = []
            for entry in split_data:
                flat_entry = {
                    'source': entry['source'],
                    'drug_id': entry['drug_id'],
                    'target_id': entry['target_id'],
                    'interaction_type': entry['interaction_type'],
                    'binding_affinity': entry['binding_affinity'],
                    'confidence_score': entry['confidence_score'],
                    'evidence_quality': entry['evidence_quality'],
                    'annotation_status': entry['annotation_status'],
                    'drug_name': entry['drug_metadata']['drug_name'],
                    'smiles': entry['drug_metadata']['smiles'],
                    'molecular_weight': entry['drug_metadata']['molecular_weight'],
                    'logp': entry['drug_metadata']['logp'],
                    'target_name': entry['target_metadata']['target_name'],
                    'uniprot_id': entry['target_metadata']['uniprot_id'],
                    'organism': entry['target_metadata']['organism'],
                    'target_class': entry['target_metadata']['target_class']
                }
                flat_data.append(flat_entry)
            
            flat_df = pd.DataFrame(flat_data)
            flat_df.to_csv(labeled_dir / f"{split_name}_dataset.csv", index=False)
    
    logger.info(f"Labeled datasets saved to: {labeled_dir}")
    
    return labeled_dir

def create_label_studio_import_files(labeled_data_all):
    """Create Label Studio import files from labeled data."""
    logger.info("Creating Label Studio import files...")
    
    label_studio_dir = Path("label_studio")
    
    # Convert labeled data to Label Studio format
    ls_tasks = []
    for idx, entry in enumerate(labeled_data_all):
        task = {
            "id": idx + 1,
            "data": {
                "drug_id": entry['drug_id'],
                "drug_name": entry['drug_metadata']['drug_name'],
                "smiles": entry['drug_metadata']['smiles'],
                "molecular_weight": entry['drug_metadata']['molecular_weight'] or 'N/A',
                "logp": entry['drug_metadata']['logp'] or 'N/A',
                "target_id": entry['target_id'],
                "target_name": entry['target_metadata']['target_name'],
                "uniprot_id": entry['target_metadata']['uniprot_id'],
                "organism": entry['target_metadata']['organism'],
                "target_class": entry['target_metadata']['target_class']
            },
            "annotations": [{
                "id": idx + 1,
                "created_username": "import_script",
                "created_ago": "imported",
                "task": idx + 1,
                "result": [
                    {
                        "value": {
                            "choices": [entry['interaction_type']]
                        },
                        "from_name": "interaction_type",
                        "to_name": "drug_id",
                        "type": "choices"
                    },
                    {
                        "value": {
                            "rating": min(5, max(1, int(entry['confidence_score'] * 5)))
                        },
                        "from_name": "confidence",
                        "to_name": "drug_id",
                        "type": "rating"
                    },
                    {
                        "value": {
                            "number": entry['binding_affinity'] if entry['binding_affinity'] else 5.0
                        },
                        "from_name": "binding_affinity",
                        "to_name": "drug_id",
                        "type": "number"
                    },
                    {
                        "value": {
                            "choices": [entry['evidence_quality']]
                        },
                        "from_name": "evidence_quality",
                        "to_name": "drug_id",
                        "type": "choices"
                    }
                ]
            }],
            "meta": {
                "source": entry['source'],
                "annotation_status": entry['annotation_status'],
                "import_date": datetime.now().isoformat()
            }
        }
        ls_tasks.append(task)
    
    # Save Label Studio tasks
    ls_file = label_studio_dir / "imported_labeled_tasks.json"
    with open(ls_file, 'w') as f:
        json.dump(ls_tasks, f, indent=2)
    
    logger.info(f"Label Studio import file created: {ls_file}")
    logger.info(f"Total tasks with annotations: {len(ls_tasks)}")
    
    return ls_file

def generate_import_summary(labeled_data_all, splits, labeled_dir):
    """Generate a comprehensive import summary."""
    logger.info("Generating import summary...")
    
    # Calculate statistics
    total_interactions = len(labeled_data_all)
    sources = {}
    interaction_types = {}
    evidence_quality = {}
    
    for entry in labeled_data_all:
        # Count by source
        source = entry['source']
        sources[source] = sources.get(source, 0) + 1
        
        # Count by interaction type
        int_type = entry['interaction_type']
        interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        # Count by evidence quality
        evidence = entry['evidence_quality']
        evidence_quality[evidence] = evidence_quality.get(evidence, 0) + 1
    
    summary = {
        "import_timestamp": datetime.now().isoformat(),
        "total_interactions": total_interactions,
        "data_sources": sources,
        "interaction_type_distribution": interaction_types,
        "evidence_quality_distribution": evidence_quality,
        "data_splits": {
            "train_size": len(splits['train']),
            "validation_size": len(splits['validation']),
            "test_size": len(splits['test'])
        },
        "files_created": {
            "labeled_data_directory": str(labeled_dir),
            "complete_dataset": str(labeled_dir / "complete_labeled_dataset.json"),
            "train_set": str(labeled_dir / "train_dataset.json"),
            "validation_set": str(labeled_dir / "validation_dataset.json"),
            "test_set": str(labeled_dir / "test_dataset.json"),
            "label_studio_import": "label_studio/imported_labeled_tasks.json"
        },
        "quality_metrics": {
            "high_quality_evidence": evidence_quality.get('high', 0),
            "medium_quality_evidence": evidence_quality.get('medium', 0),
            "low_quality_evidence": evidence_quality.get('low', 0),
            "pre_labeled_count": sum(1 for entry in labeled_data_all if entry['annotation_status'] == 'pre_labeled'),
            "verified_count": sum(1 for entry in labeled_data_all if entry['annotation_status'] == 'verified')
        }
    }
    
    # Save summary
    summary_file = labeled_dir / "import_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary, summary_file

def main():
    """Main import function."""
    logger.info("Starting labeled dataset import process...")
    
    try:
        # Load parameters
        params = load_params()
        
        # Import data from different sources
        all_labeled_data = []
        
        # Import from different sources
        biosnap_data = import_biosnap_data()
        bindingdb_data = import_bindingdb_data()
        sample_data = import_sample_data()
        
        # Combine all data
        all_labeled_data.extend(biosnap_data)
        all_labeled_data.extend(bindingdb_data)
        all_labeled_data.extend(sample_data)
        
        if not all_labeled_data:
            logger.error("No labeled data was imported successfully")
            return 1
        
        # Create train/validation/test splits
        splits = create_train_test_splits(all_labeled_data, params)
        
        # Save labeled datasets
        labeled_dir = save_labeled_datasets(splits, all_labeled_data)
        
        # Create Label Studio import files
        ls_file = create_label_studio_import_files(all_labeled_data)
        
        # Generate summary
        summary, summary_file = generate_import_summary(all_labeled_data, splits, labeled_dir)
        
        # Print results
        print("\n📊 Labeled Dataset Import Complete!")
        print("=" * 50)
        print(f"📈 Total Interactions: {summary['total_interactions']}")
        print(f"📂 Data Sources: {summary['data_sources']}")
        print(f"🏷️  Interaction Types: {summary['interaction_type_distribution']}")
        print(f"📋 Evidence Quality: {summary['evidence_quality_distribution']}")
        print()
        print("📊 Data Splits:")
        print(f"  Training: {summary['data_splits']['train_size']}")
        print(f"  Validation: {summary['data_splits']['validation_size']}")
        print(f"  Test: {summary['data_splits']['test_size']}")
        print()
        print("📁 Files Created:")
        print(f"  📊 Labeled Data: {labeled_dir}")
        print(f"  🏷️  Label Studio Import: {ls_file}")
        print(f"  📋 Summary Report: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Labeled dataset import failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())