#!/usr/bin/env python3
"""
Great Expectations Setup Script for MLOps Drug Repurposing Project

This script sets up Great Expectations data validation suites for the drug repurposing datasets.
"""

import os
import pandas as pd
from pathlib import Path
import great_expectations as ge
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig
from great_expectations.core.batch import RuntimeBatchRequest
import logging

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_context():
    """Set up Great Expectations data context."""
    context_root_dir = Path("great_expectations")
    context_root_dir.mkdir(exist_ok=True)
    
    # Create data context configuration
    data_context_config = DataContextConfig(
        config_version=3.0,
        datasources={
            "mlops_drug_data": {
                "class_name": "Datasource",
                "execution_engine": {
                    "class_name": "PandasExecutionEngine"
                },
                "data_connectors": {
                    "default_runtime_data_connector_name": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"]
                    },
                    "default_inferred_data_connector_name": {
                        "class_name": "InferredAssetFilesystemDataConnector",
                        "base_directory": "../data/raw/sample/",
                        "default_regex": {
                            "group_names": ["data_asset_name"],
                            "pattern": r"(.*)\.csv"
                        }
                    }
                }
            }
        },
        stores={
            "expectations_store": {
                "class_name": "ExpectationsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "expectations/"
                }
            },
            "validations_store": {
                "class_name": "ValidationsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "uncommitted/validations/"
                }
            },
            "evaluation_parameter_store": {
                "class_name": "EvaluationParameterStore"
            },
            "checkpoint_store": {
                "class_name": "CheckpointStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "checkpoints/"
                }
            }
        },
        data_docs_sites={
            "local_site": {
                "class_name": "SiteBuilder",
                "show_how_to_buttons": True,
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "uncommitted/data_docs/local_site/"
                },
                "site_index_builder": {
                    "class_name": "DefaultSiteIndexBuilder"
                }
            }
        },
        expectations_store_name="expectations_store",
        validations_store_name="validations_store",
        evaluation_parameter_store_name="evaluation_parameter_store",
        checkpoint_store_name="checkpoint_store",
        anonymous_usage_statistics={
            "enabled": False,
            "data_context_id": "mlops-drug-repurposing-context"
        }
    )
    
    # Initialize context
    context = BaseDataContext(project_config=data_context_config, context_root_dir=context_root_dir)
    
    return context

def create_drug_target_interactions_suite(context):
    """Create expectation suite for drug-target interactions dataset."""
    suite_name = "drug_target_interactions_suite"
    
    try:
        suite = context.get_expectation_suite(suite_name)
        logger.info(f"Using existing suite: {suite_name}")
    except:
        suite = context.create_expectation_suite(suite_name)
        logger.info(f"Created new suite: {suite_name}")
    
    # Load sample data
    data_path = Path("data/raw/sample/drug_target_interactions.csv")
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return suite
    
    df = pd.read_csv(data_path)
    validator = context.get_validator(
        batch_request=RuntimeBatchRequest(
            datasource_name="mlops_drug_data",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="drug_target_interactions",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "drug_target_interactions"}
        ),
        expectation_suite_name=suite_name
    )
    
    # Add expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["drug_id", "target_id", "interaction_type"]
    )
    
    validator.expect_table_row_count_to_be_between(min_value=100, max_value=100000)
    
    validator.expect_column_values_to_not_be_null("drug_id")
    validator.expect_column_values_to_not_be_null("target_id")
    validator.expect_column_values_to_not_be_null("interaction_type")
    
    validator.expect_column_values_to_be_of_type("drug_id", "object")
    validator.expect_column_values_to_be_of_type("target_id", "object")
    
    validator.expect_column_values_to_be_in_set(
        "interaction_type", 
        ["binding", "inhibition", "activation", "unknown"]
    )
    
    # Check for reasonable number of unique drugs and targets
    validator.expect_column_unique_value_count_to_be_between("drug_id", min_value=10, max_value=10000)
    validator.expect_column_unique_value_count_to_be_between("target_id", min_value=5, max_value=5000)
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    return suite

def create_drug_metadata_suite(context):
    """Create expectation suite for drug metadata dataset."""
    suite_name = "drug_metadata_suite"
    
    try:
        suite = context.get_expectation_suite(suite_name)
        logger.info(f"Using existing suite: {suite_name}")
    except:
        suite = context.create_expectation_suite(suite_name)
        logger.info(f"Created new suite: {suite_name}")
    
    # Load sample data
    data_path = Path("data/raw/sample/drug_metadata.csv")
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return suite
    
    df = pd.read_csv(data_path)
    validator = context.get_validator(
        batch_request=RuntimeBatchRequest(
            datasource_name="mlops_drug_data",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="drug_metadata",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "drug_metadata"}
        ),
        expectation_suite_name=suite_name
    )
    
    # Add expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["drug_id", "drug_name", "smiles"]
    )
    
    validator.expect_table_row_count_to_be_between(min_value=10, max_value=50000)
    
    validator.expect_column_values_to_not_be_null("drug_id")
    validator.expect_column_values_to_not_be_null("drug_name")
    validator.expect_column_values_to_not_be_null("smiles", mostly=0.95)
    
    validator.expect_column_values_to_be_unique("drug_id")
    
    # SMILES validation
    validator.expect_column_value_lengths_to_be_between("smiles", min_value=5, max_value=500)
    validator.expect_column_values_to_match_regex(
        "smiles", 
        regex=r"^[A-Za-z0-9@+\-\[\]()=#$\\.\\\\]+$",
        mostly=0.90
    )
    
    # Drug name validation
    validator.expect_column_value_lengths_to_be_between("drug_name", min_value=2, max_value=100)
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    return suite

def create_target_metadata_suite(context):
    """Create expectation suite for target metadata dataset."""
    suite_name = "target_metadata_suite"
    
    try:
        suite = context.get_expectation_suite(suite_name)
        logger.info(f"Using existing suite: {suite_name}")
    except:
        suite = context.create_expectation_suite(suite_name)
        logger.info(f"Created new suite: {suite_name}")
    
    # Load sample data
    data_path = Path("data/raw/sample/target_metadata.csv")
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return suite
    
    df = pd.read_csv(data_path)
    validator = context.get_validator(
        batch_request=RuntimeBatchRequest(
            datasource_name="mlops_drug_data",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name="target_metadata",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "target_metadata"}
        ),
        expectation_suite_name=suite_name
    )
    
    # Add expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["target_id", "target_name", "uniprot_id"]
    )
    
    validator.expect_table_row_count_to_be_between(min_value=5, max_value=10000)
    
    validator.expect_column_values_to_not_be_null("target_id")
    validator.expect_column_values_to_not_be_null("target_name")
    validator.expect_column_values_to_not_be_null("uniprot_id", mostly=0.90)
    
    validator.expect_column_values_to_be_unique("target_id")
    
    # UniProt ID validation
    validator.expect_column_values_to_match_regex(
        "uniprot_id", 
        regex=r"^[A-Z0-9]{6,10}$",
        mostly=0.85
    )
    
    # Target name validation
    validator.expect_column_value_lengths_to_be_between("target_name", min_value=2, max_value=100)
    
    # Save suite
    validator.save_expectation_suite(discard_failed_expectations=False)
    
    return suite

def create_data_validation_checkpoint(context):
    """Create a checkpoint for automated data validation."""
    checkpoint_name = "drug_repurposing_data_checkpoint"
    
    checkpoint_config = {
        "name": checkpoint_name,
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-drug-repurposing-validation",
        "validations": [
            {
                "batch_request": {
                    "datasource_name": "mlops_drug_data",
                    "data_connector_name": "default_inferred_data_connector_name",
                    "data_asset_name": "drug_target_interactions",
                    "data_connector_query": {"index": -1}
                },
                "expectation_suite_name": "drug_target_interactions_suite"
            },
            {
                "batch_request": {
                    "datasource_name": "mlops_drug_data",
                    "data_connector_name": "default_inferred_data_connector_name",
                    "data_asset_name": "drug_metadata",
                    "data_connector_query": {"index": -1}
                },
                "expectation_suite_name": "drug_metadata_suite"
            },
            {
                "batch_request": {
                    "datasource_name": "mlops_drug_data",
                    "data_connector_name": "default_inferred_data_connector_name",
                    "data_asset_name": "target_metadata",
                    "data_connector_query": {"index": -1}
                },
                "expectation_suite_name": "target_metadata_suite"
            }
        ],
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {"class_name": "StoreValidationResultAction"}
            },
            {
                "name": "update_data_docs",
                "action": {"class_name": "UpdateDataDocsAction", "site_names": ["local_site"]}
            }
        ]
    }
    
    try:
        context.add_checkpoint(**checkpoint_config)
        logger.info(f"Created checkpoint: {checkpoint_name}")
    except Exception as e:
        logger.info(f"Checkpoint {checkpoint_name} already exists or error: {e}")
    
    return checkpoint_name

def main():
    """Main setup function."""
    logger.info("Setting up Great Expectations for MLOps Drug Repurposing Project...")
    
    try:
        # Setup data context
        context = setup_data_context()
        logger.info("Data context created successfully")
        
        # Create expectation suites
        create_drug_target_interactions_suite(context)
        create_drug_metadata_suite(context)
        create_target_metadata_suite(context)
        
        # Create checkpoint
        checkpoint_name = create_data_validation_checkpoint(context)
        
        # Build data docs
        context.build_data_docs()
        
        logger.info("Great Expectations setup completed successfully!")
        logger.info(f"Checkpoint created: {checkpoint_name}")
        logger.info("Data docs built - check great_expectations/uncommitted/data_docs/local_site/index.html")
        
        return 0
        
    except Exception as e:
        logger.error(f"Great Expectations setup failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())