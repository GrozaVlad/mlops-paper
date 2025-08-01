#!/usr/bin/env python3
"""
Data Validation Script for MLOps Drug Repurposing Project

This script validates the downloaded datasets and generates validation reports.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig
import mlflow
import os

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def validate_dataset_structure(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> Dict[str, Any]:
    """Validate dataset structure and basic properties."""
    validation_results = {
        "dataset_name": dataset_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_columns": [],
        "extra_columns": [],
        "missing_data_percentage": {},
        "data_types": {},
        "validation_passed": True,
        "issues": []
    }
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        validation_results["missing_columns"] = list(missing_columns)
        validation_results["validation_passed"] = False
        validation_results["issues"].append(f"Missing required columns: {missing_columns}")
    
    # Check for extra columns
    extra_columns = set(df.columns) - set(required_columns)
    if extra_columns:
        validation_results["extra_columns"] = list(extra_columns)
        logger.info(f"Extra columns found in {dataset_name}: {extra_columns}")
    
    # Check missing data
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        validation_results["missing_data_percentage"][col] = round(float(missing_pct), 2)
        validation_results["data_types"][col] = str(df[col].dtype)
        
        # Flag high missing data
        if missing_pct > 50:
            validation_results["validation_passed"] = False
            validation_results["issues"].append(f"Column {col} has {missing_pct:.1f}% missing data")
    
    return validation_results

def validate_drug_target_interactions(df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    """Validate drug-target interaction dataset."""
    logger.info("Validating drug-target interactions dataset...")
    
    required_columns = params["validation"]["required_columns"]["drug_target_interactions"]
    validation_results = validate_dataset_structure(df, required_columns, "drug_target_interactions")
    
    if not validation_results["validation_passed"]:
        return validation_results
    
    # Specific validations for drug-target interactions
    # Check for duplicate interactions
    if "drug_id" in df.columns and "target_id" in df.columns:
        duplicates = df[["drug_id", "target_id"]].duplicated().sum()
        validation_results["duplicate_interactions"] = int(duplicates)
        if duplicates > 0:
            validation_results["issues"].append(f"Found {duplicates} duplicate drug-target pairs")
    
    # Check interaction types
    if "interaction_type" in df.columns:
        unique_types = df["interaction_type"].nunique()
        validation_results["unique_interaction_types"] = int(unique_types)
        validation_results["interaction_type_counts"] = df["interaction_type"].value_counts().to_dict()
    
    # Check binding affinity values
    if "binding_affinity" in df.columns:
        affinity_stats = {
            "min": float(df["binding_affinity"].min()),
            "max": float(df["binding_affinity"].max()),
            "mean": float(df["binding_affinity"].mean()),
            "std": float(df["binding_affinity"].std())
        }
        validation_results["binding_affinity_stats"] = affinity_stats
        
        # Check for unrealistic values
        if affinity_stats["min"] < 0 or affinity_stats["max"] > 15:
            validation_results["issues"].append("Binding affinity values outside expected range (0-15)")
    
    return validation_results

def validate_smiles_chemistry(smiles_series: pd.Series) -> Dict[str, Any]:
    """Validate SMILES strings using RDKit for chemical validity."""
    valid_molecules = 0
    invalid_molecules = 0
    molecular_properties = []
    
    for smiles in smiles_series.dropna():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules += 1
                # Calculate molecular properties
                props = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol)
                }
                molecular_properties.append(props)
            else:
                invalid_molecules += 1
        except:
            invalid_molecules += 1
    
    molecular_props_df = pd.DataFrame(molecular_properties) if molecular_properties else pd.DataFrame()
    
    return {
        'valid_molecules': valid_molecules,
        'invalid_molecules': invalid_molecules,
        'validity_rate': valid_molecules / (valid_molecules + invalid_molecules) if (valid_molecules + invalid_molecules) > 0 else 0,
        'molecular_properties_summary': molecular_props_df.describe().to_dict() if not molecular_props_df.empty else {}
    }

def validate_drug_metadata(df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    """Validate drug metadata dataset with enhanced chemical validation."""
    logger.info("Validating drug metadata dataset...")
    
    required_columns = params["validation"]["required_columns"]["drug_metadata"]
    validation_results = validate_dataset_structure(df, required_columns, "drug_metadata")
    
    if not validation_results["validation_passed"]:
        return validation_results
    
    # Enhanced SMILES validation using RDKit
    if "smiles" in df.columns:
        chemistry_validation = validate_smiles_chemistry(df["smiles"])
        validation_results.update(chemistry_validation)
        
        if chemistry_validation['validity_rate'] < 0.95:
            validation_results["issues"].append(
                f"SMILES validity rate {chemistry_validation['validity_rate']:.2%} below threshold (95%)"
            )
        
        # Analyze molecular properties if available
        if chemistry_validation['molecular_properties_summary']:
            validation_results["molecular_properties_stats"] = chemistry_validation['molecular_properties_summary']
    
    # Check molecular weight range (if provided separately)
    if "molecular_weight" in df.columns:
        mw_stats = {
            "min": float(df["molecular_weight"].min()),
            "max": float(df["molecular_weight"].max()),
            "mean": float(df["molecular_weight"].mean())
        }
        validation_results["molecular_weight_stats"] = mw_stats
        
        # Check for unrealistic molecular weights
        if mw_stats["min"] < 50 or mw_stats["max"] > 2000:
            validation_results["issues"].append("Molecular weights outside typical drug range (50-2000 Da)")
    
    return validation_results

def validate_target_metadata(df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    """Validate target metadata dataset."""
    logger.info("Validating target metadata dataset...")
    
    required_columns = params["validation"]["required_columns"]["target_metadata"]
    validation_results = validate_dataset_structure(df, required_columns, "target_metadata")
    
    if not validation_results["validation_passed"]:
        return validation_results
    
    # Specific validations for target metadata
    # Check UniProt ID format
    if "uniprot_id" in df.columns:
        valid_uniprot = df["uniprot_id"].str.match(r'^[A-Z0-9]{6,10}$', na=False)
        invalid_uniprot_count = (~valid_uniprot).sum()
        validation_results["invalid_uniprot_count"] = int(invalid_uniprot_count)
        if invalid_uniprot_count > 0:
            validation_results["issues"].append(f"Found {invalid_uniprot_count} invalid UniProt IDs")
    
    # Check target classes
    if "target_class" in df.columns:
        target_classes = df["target_class"].value_counts().to_dict()
        validation_results["target_class_distribution"] = target_classes
    
    return validation_results

def run_great_expectations_validation(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Run Great Expectations validation suite on dataset."""
    try:
        # Convert pandas DataFrame to Great Expectations Dataset
        ge_df = ge.from_pandas(df)
        
        # Define expectations based on dataset type
        expectations_results = {}
        
        if dataset_name == "drug_target_interactions":
            # Drug-target interaction expectations
            expectations_results['expect_table_row_count_to_be_between'] = ge_df.expect_table_row_count_to_be_between(min_value=100)
            if 'drug_id' in df.columns:
                expectations_results['expect_column_to_exist_drug_id'] = ge_df.expect_column_to_exist('drug_id')
                expectations_results['expect_column_values_to_not_be_null_drug_id'] = ge_df.expect_column_values_to_not_be_null('drug_id')
            if 'target_id' in df.columns:
                expectations_results['expect_column_to_exist_target_id'] = ge_df.expect_column_to_exist('target_id')
                expectations_results['expect_column_values_to_not_be_null_target_id'] = ge_df.expect_column_values_to_not_be_null('target_id')
            
        elif dataset_name == "drug_metadata":
            # Drug metadata expectations
            expectations_results['expect_table_row_count_to_be_between'] = ge_df.expect_table_row_count_to_be_between(min_value=50)
            if 'smiles' in df.columns:
                expectations_results['expect_column_values_to_not_be_null_smiles'] = ge_df.expect_column_values_to_not_be_null('smiles', mostly=0.95)
                expectations_results['expect_column_value_lengths_to_be_between_smiles'] = ge_df.expect_column_value_lengths_to_be_between('smiles', min_value=5, max_value=500)
            
        elif dataset_name == "target_metadata":
            # Target metadata expectations
            expectations_results['expect_table_row_count_to_be_between'] = ge_df.expect_table_row_count_to_be_between(min_value=10)
            if 'uniprot_id' in df.columns:
                expectations_results['expect_column_values_to_match_regex_uniprot'] = ge_df.expect_column_values_to_match_regex('uniprot_id', regex=r'^[A-Z0-9]{6,10}$', mostly=0.9)
        
        # Extract results
        ge_summary = {
            'total_expectations': len(expectations_results),
            'successful_expectations': sum(1 for result in expectations_results.values() if result.success),
            'success_rate': sum(1 for result in expectations_results.values() if result.success) / len(expectations_results) if expectations_results else 0,
            'expectations_details': {key: {'success': result.success, 'result': result.result} for key, result in expectations_results.items()}
        }
        
        return ge_summary
        
    except Exception as e:
        logger.warning(f"Great Expectations validation failed for {dataset_name}: {e}")
        return {'error': str(e), 'total_expectations': 0, 'successful_expectations': 0, 'success_rate': 0}

def create_validation_visualizations(validation_results: List[Dict], output_dir: Path) -> None:
    """Create visualization plots for validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Data Quality Score by Dataset
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Quality scores
    datasets = [result['dataset_name'] for result in validation_results]
    quality_scores = []
    
    for result in validation_results:
        issues_count = len(result['issues'])
        avg_missing_data = np.mean(list(result['missing_data_percentage'].values()))
        quality_score = max(0, 100 - (issues_count * 10) - avg_missing_data)
        quality_scores.append(quality_score)
    
    axes[0, 0].bar(datasets, quality_scores, color=['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in quality_scores])
    axes[0, 0].set_title('Data Quality Scores by Dataset')
    axes[0, 0].set_ylabel('Quality Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Missing Data Heatmap
    missing_data = {}
    for result in validation_results:
        missing_data[result['dataset_name']] = result['missing_data_percentage']
    
    if missing_data:
        # Create combined missing data DataFrame
        all_columns = set()
        for dataset_missing in missing_data.values():
            all_columns.update(dataset_missing.keys())
        
        missing_df = pd.DataFrame(index=list(missing_data.keys()), columns=list(all_columns))
        for dataset, missing_cols in missing_data.items():
            for col, pct in missing_cols.items():
                missing_df.loc[dataset, col] = pct
        
        missing_df = missing_df.fillna(0).astype(float)
        
        if not missing_df.empty:
            sns.heatmap(missing_df, annot=True, fmt='.1f', cmap='Reds', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Data Percentage by Dataset and Column')
    
    # 3. Issues Count
    issue_counts = [len(result['issues']) for result in validation_results]
    axes[1, 0].bar(datasets, issue_counts, color=['red' if count > 0 else 'green' for count in issue_counts])
    axes[1, 0].set_title('Validation Issues Count by Dataset')
    axes[1, 0].set_ylabel('Number of Issues')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Dataset Sizes
    row_counts = [result['total_rows'] for result in validation_results]
    axes[1, 1].bar(datasets, row_counts, color='skyblue')
    axes[1, 1].set_title('Dataset Sizes (Number of Rows)')
    axes[1, 1].set_ylabel('Number of Rows')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_validation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Validation visualizations saved to {output_dir}")

def generate_validation_metrics(validation_results: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive summary metrics from validation results."""
    metrics = {
        "validation_timestamp": datetime.now().isoformat(),
        "total_datasets": len(validation_results),
        "datasets_passed": sum(1 for result in validation_results if result["validation_passed"]),
        "total_issues": sum(len(result["issues"]) for result in validation_results),
        "overall_validation_passed": all(result["validation_passed"] for result in validation_results)
    }
    
    # Calculate quality scores and detailed metrics
    dataset_metrics = {}
    for result in validation_results:
        dataset_name = result["dataset_name"]
        total_columns = result["total_columns"]
        issues_count = len(result["issues"])
        
        # Enhanced quality score calculation
        avg_missing_data = np.mean(list(result["missing_data_percentage"].values()))
        completeness_score = 100 - avg_missing_data
        validity_score = max(0, 100 - (issues_count * 15))
        
        # Great Expectations score if available
        ge_score = 100
        if 'great_expectations' in result:
            ge_score = result['great_expectations']['success_rate'] * 100
        
        # Overall quality score (weighted average)
        quality_score = (completeness_score * 0.4 + validity_score * 0.4 + ge_score * 0.2)
        
        dataset_metrics[dataset_name] = {
            'quality_score': round(quality_score, 2),
            'completeness_score': round(completeness_score, 2),
            'validity_score': round(validity_score, 2),
            'great_expectations_score': round(ge_score, 2),
            'total_rows': result['total_rows'],
            'total_columns': total_columns,
            'issues_count': issues_count,
            'missing_data_avg': round(avg_missing_data, 2)
        }
        
        # Add to flat metrics for DVC compatibility
        metrics[f"{dataset_name}_quality_score"] = round(quality_score, 2)
        metrics[f"{dataset_name}_total_rows"] = result["total_rows"]
        metrics[f"{dataset_name}_issues_count"] = issues_count
        metrics[f"{dataset_name}_completeness_score"] = round(completeness_score, 2)
    
    metrics['dataset_details'] = dataset_metrics
    
    # Calculate overall project health score
    if dataset_metrics:
        overall_quality = np.mean([details['quality_score'] for details in dataset_metrics.values()])
        metrics['overall_quality_score'] = round(overall_quality, 2)
        
        # Determine data readiness level
        if overall_quality >= 95:
            metrics['data_readiness_level'] = 'PRODUCTION_READY'
        elif overall_quality >= 85:
            metrics['data_readiness_level'] = 'STAGING_READY'
        elif overall_quality >= 70:
            metrics['data_readiness_level'] = 'DEVELOPMENT_READY'
        else:
            metrics['data_readiness_level'] = 'NEEDS_IMPROVEMENT'
    
    return metrics

def log_to_mlflow(validation_results: List[Dict], metrics: Dict[str, Any]) -> None:
    """Log validation results and metrics to MLflow."""
    try:
        mlflow.set_experiment("data_validation")
        
        with mlflow.start_run(run_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, value)
                elif isinstance(value, str):
                    mlflow.log_param(key, value)
            
            # Log validation results as artifacts
            with open('temp_validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)
            mlflow.log_artifact('temp_validation_results.json', 'validation_results')
            
            # Clean up temp file
            os.remove('temp_validation_results.json')
            
            logger.info("Validation results logged to MLflow")
            
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")

def main():
    """Enhanced main validation function with comprehensive data quality checks."""
    logger.info("Starting comprehensive data validation pipeline...")
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Load parameters
    params = load_params()
    
    # Define data paths
    data_dir = Path("data/raw/sample")
    output_dir = Path("data/validation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = []
    
    try:
        # Validate drug-target interactions
        interactions_file = data_dir / "drug_target_interactions.csv"
        if interactions_file.exists():
            logger.info("Validating drug-target interactions...")
            df_interactions = pd.read_csv(interactions_file)
            interactions_result = validate_drug_target_interactions(df_interactions, params)
            
            # Add Great Expectations validation
            ge_result = run_great_expectations_validation(df_interactions, "drug_target_interactions")
            interactions_result['great_expectations'] = ge_result
            
            validation_results.append(interactions_result)
        else:
            logger.warning(f"Drug-target interactions file not found: {interactions_file}")
        
        # Validate drug metadata
        drugs_file = data_dir / "drug_metadata.csv"
        if drugs_file.exists():
            logger.info("Validating drug metadata...")
            df_drugs = pd.read_csv(drugs_file)
            drugs_result = validate_drug_metadata(df_drugs, params)
            
            # Add Great Expectations validation
            ge_result = run_great_expectations_validation(df_drugs, "drug_metadata")
            drugs_result['great_expectations'] = ge_result
            
            validation_results.append(drugs_result)
        else:
            logger.warning(f"Drug metadata file not found: {drugs_file}")
        
        # Validate target metadata
        targets_file = data_dir / "target_metadata.csv"
        if targets_file.exists():
            logger.info("Validating target metadata...")
            df_targets = pd.read_csv(targets_file)
            targets_result = validate_target_metadata(df_targets, params)
            
            # Add Great Expectations validation
            ge_result = run_great_expectations_validation(df_targets, "target_metadata")
            targets_result['great_expectations'] = ge_result
            
            validation_results.append(targets_result)
        else:
            logger.warning(f"Target metadata file not found: {targets_file}")
        
        # Generate comprehensive validation report
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_version": "2.0",
            "validation_results": validation_results,
            "summary": {
                "total_datasets_validated": len(validation_results),
                "datasets_passed": sum(1 for result in validation_results if result["validation_passed"]),
                "overall_status": "PASSED" if all(result["validation_passed"] for result in validation_results) else "FAILED",
                "validation_framework": "Custom + Great Expectations"
            }
        }
        
        # Generate comprehensive metrics
        metrics = generate_validation_metrics(validation_results)
        
        # Create visualizations
        create_validation_visualizations(validation_results, output_dir)
        
        # Save validation report
        with open("data/validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)
        
        # Save metrics
        with open("data/validation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Log to MLflow
        log_to_mlflow(validation_results, metrics)
        
        # Enhanced reporting
        logger.info(f"Validation completed. Status: {validation_report['summary']['overall_status']}")
        logger.info(f"Overall Quality Score: {metrics.get('overall_quality_score', 'N/A')}")
        logger.info(f"Data Readiness Level: {metrics.get('data_readiness_level', 'N/A')}")
        logger.info(f"Results saved to: data/validation_report.json")
        logger.info(f"Metrics saved to: data/validation_metrics.json")
        logger.info(f"Visualizations saved to: {output_dir}")
        
        # Print detailed summary
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Quality Score: {metrics.get('overall_quality_score', 'N/A')}/100")
        print(f"Data Readiness Level: {metrics.get('data_readiness_level', 'N/A')}")
        print(f"Total Datasets: {len(validation_results)}")
        print()
        
        for result in validation_results:
            dataset_name = result["dataset_name"]
            status = "✅ PASSED" if result["validation_passed"] else "❌ FAILED"
            quality_score = metrics['dataset_details'].get(dataset_name, {}).get('quality_score', 'N/A')
            
            print(f"{dataset_name}: {status} (Quality: {quality_score}/100)")
            
            # Great Expectations summary
            if 'great_expectations' in result:
                ge_success_rate = result['great_expectations']['success_rate'] * 100
                print(f"  GE Validation: {ge_success_rate:.1f}% expectations passed")
            
            if result["issues"]:
                print("  Issues:")
                for issue in result["issues"]:
                    print(f"    - {issue}")
            print()
        
        print("="*60)
        
        return 0 if validation_report['summary']['overall_status'] == "PASSED" else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())