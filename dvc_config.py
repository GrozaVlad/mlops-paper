#!/usr/bin/env python3
"""
DVC Configuration Script for MLOps Drug Repurposing Project

This script configures DVC with S3 remote storage based on environment.
It reads the Terraform outputs to get the appropriate S3 bucket names.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return None

def get_terraform_outputs(terraform_dir="terraform"):
    """Get Terraform outputs to retrieve S3 bucket names."""
    print("Getting Terraform outputs...")
    
    # Check if terraform directory exists
    if not Path(terraform_dir).exists():
        print(f"Terraform directory {terraform_dir} not found.")
        print("Please run terraform apply first.")
        return None
    
    # Get terraform outputs
    cmd = f"cd {terraform_dir} && terraform output -json"
    output = run_command(cmd, check=False)
    
    if output:
        try:
            outputs = json.loads(output)
            return outputs
        except json.JSONDecodeError:
            print("Failed to parse terraform outputs")
            return None
    else:
        print("No terraform outputs found. Using default bucket names.")
        return None

def configure_dvc_remote(environment="dev"):
    """Configure DVC remote storage based on environment."""
    
    # Get terraform outputs
    tf_outputs = get_terraform_outputs()
    
    if tf_outputs:
        # Use actual bucket names from terraform
        data_bucket = tf_outputs.get("s3_data_bucket_name", {}).get("value")
        models_bucket = tf_outputs.get("s3_models_bucket_name", {}).get("value") 
        artifacts_bucket = tf_outputs.get("s3_artifacts_bucket_name", {}).get("value")
    else:
        # Use expected bucket names if terraform not available
        project_name = "mlops-drug-repurposing"
        data_bucket = f"{project_name}-{environment}-data-randomsuffix"
        models_bucket = f"{project_name}-{environment}-models-randomsuffix"
        artifacts_bucket = f"{project_name}-{environment}-artifacts-randomsuffix"
        
        print(f"Using default bucket names for environment: {environment}")
        print("Note: You'll need to update these with actual bucket names after terraform apply")
    
    # Configure DVC remotes
    print(f"Configuring DVC remotes for environment: {environment}")
    
    # Data remote (default)
    if data_bucket:
        print(f"Setting up data remote: s3://{data_bucket}")
        run_command(f"dvc remote add -d data s3://{data_bucket}")
    
    # Models remote
    if models_bucket:
        print(f"Setting up models remote: s3://{models_bucket}")
        run_command(f"dvc remote add models s3://{models_bucket}")
    
    # Artifacts remote  
    if artifacts_bucket:
        print(f"Setting up artifacts remote: s3://{artifacts_bucket}")
        run_command(f"dvc remote add artifacts s3://{artifacts_bucket}")
    
    # Configure AWS credentials (will use AWS CLI credentials)
    print("Configuring AWS credentials for DVC...")
    run_command("dvc remote modify data credentialpath ~/.aws/credentials")
    run_command("dvc remote modify models credentialpath ~/.aws/credentials")
    run_command("dvc remote modify artifacts credentialpath ~/.aws/credentials")
    
    print("✅ DVC remote configuration completed!")
    print("\nNext steps:")
    print("1. Ensure AWS CLI is configured with appropriate credentials")
    print("2. Run 'dvc remote list' to verify remote configuration")
    print("3. Test connection with 'dvc remote list'")

def main():
    """Main function to configure DVC."""
    
    # Get environment from command line argument or default to dev
    environment = sys.argv[1] if len(sys.argv) > 1 else "dev"
    
    if environment not in ["dev", "staging", "prod"]:
        print("Error: Environment must be one of: dev, staging, prod")
        sys.exit(1)
    
    print(f"Configuring DVC for environment: {environment}")
    
    # Check if DVC is initialized
    if not Path(".dvc").exists():
        print("DVC not initialized. Initializing...")
        run_command("dvc init")
    
    # Configure DVC remote
    configure_dvc_remote(environment)

if __name__ == "__main__":
    main()