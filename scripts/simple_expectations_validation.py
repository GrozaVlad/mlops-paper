#!/usr/bin/env python3
"""
Simple Great Expectations Validation for MLOps Drug Repurposing Project

This script runs basic data quality checks using Great Expectations.
"""

import os
import pandas as pd
from pathlib import Path
import great_expectations as ge
import logging
import json

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_with_great_expectations():
    """Run simple Great Expectations validation."""
    logger.info("Running Great Expectations validation...")
    
    validation_results = {}
    
    try:
        data_dir = Path("data/raw/sample")
        
        # 1. Validate drug-target interactions
        interactions_file = data_dir / "drug_target_interactions.csv"
        if interactions_file.exists():
            logger.info("Validating drug-target interactions with GE...")
            df = pd.read_csv(interactions_file)
            ge_df = ge.from_pandas(df)
            
            results = {}
            results['row_count'] = ge_df.expect_table_row_count_to_be_between(min_value=100)
            results['drug_id_not_null'] = ge_df.expect_column_values_to_not_be_null('drug_id')
            results['target_id_not_null'] = ge_df.expect_column_values_to_not_be_null('target_id')
            results['interaction_type_not_null'] = ge_df.expect_column_values_to_not_be_null('interaction_type')
            
            validation_results['drug_target_interactions'] = {
                'total_expectations': len(results),
                'successful_expectations': sum(1 for r in results.values() if r.success),
                'success_rate': sum(1 for r in results.values() if r.success) / len(results),
                'details': {k: {'success': v.success, 'result': str(v.result)} for k, v in results.items()}
            }
        
        # 2. Validate drug metadata
        drugs_file = data_dir / "drug_metadata.csv"
        if drugs_file.exists():
            logger.info("Validating drug metadata with GE...")
            df = pd.read_csv(drugs_file)
            ge_df = ge.from_pandas(df)
            
            results = {}
            results['row_count'] = ge_df.expect_table_row_count_to_be_between(min_value=10)
            results['drug_id_not_null'] = ge_df.expect_column_values_to_not_be_null('drug_id')
            results['drug_id_unique'] = ge_df.expect_column_values_to_be_unique('drug_id')
            results['smiles_not_null'] = ge_df.expect_column_values_to_not_be_null('smiles', mostly=0.95)
            results['smiles_length'] = ge_df.expect_column_value_lengths_to_be_between('smiles', min_value=5, max_value=500)
            
            validation_results['drug_metadata'] = {
                'total_expectations': len(results),
                'successful_expectations': sum(1 for r in results.values() if r.success),
                'success_rate': sum(1 for r in results.values() if r.success) / len(results),
                'details': {k: {'success': v.success, 'result': str(v.result)} for k, v in results.items()}
            }
        
        # 3. Validate target metadata
        targets_file = data_dir / "target_metadata.csv"
        if targets_file.exists():
            logger.info("Validating target metadata with GE...")
            df = pd.read_csv(targets_file)
            ge_df = ge.from_pandas(df)
            
            results = {}
            results['row_count'] = ge_df.expect_table_row_count_to_be_between(min_value=5)
            results['target_id_not_null'] = ge_df.expect_column_values_to_not_be_null('target_id')
            results['target_id_unique'] = ge_df.expect_column_values_to_be_unique('target_id')
            results['uniprot_format'] = ge_df.expect_column_values_to_match_regex('uniprot_id', 
                                                                                 regex=r'^[A-Z0-9]{6,10}$', 
                                                                                 mostly=0.8)
            
            validation_results['target_metadata'] = {
                'total_expectations': len(results),
                'successful_expectations': sum(1 for r in results.values() if r.success),
                'success_rate': sum(1 for r in results.values() if r.success) / len(results),
                'details': {k: {'success': v.success, 'result': str(v.result)} for k, v in results.items()}
            }
        
        # Save results
        output_file = Path("data/great_expectations_validation.json")
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Print summary
        print("\n🔍 Great Expectations Validation Summary")
        print("=" * 50)
        
        for dataset, results in validation_results.items():
            success_rate = results['success_rate'] * 100
            status = "✅ PASSED" if success_rate >= 80 else "⚠️ WARNING" if success_rate >= 60 else "❌ FAILED"
            
            print(f"{dataset}: {status}")
            print(f"  Success Rate: {success_rate:.1f}% ({results['successful_expectations']}/{results['total_expectations']})")
            
            failed_expectations = [k for k, v in results['details'].items() if not v['success']]
            if failed_expectations:
                print(f"  Failed: {', '.join(failed_expectations)}")
            print()
        
        logger.info(f"Great Expectations validation results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Great Expectations validation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(validate_with_great_expectations())