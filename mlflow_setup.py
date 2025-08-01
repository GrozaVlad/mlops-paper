#!/usr/bin/env python3
"""
MLflow Setup Script for MLOps Drug Repurposing Project

This script configures MLflow tracking server and creates initial experiments.
"""

import os
import mlflow
import mlflow.tracking
from pathlib import Path
import subprocess
import sys
import yaml

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_mlflow_tracking(environment="dev"):
    """Set up MLflow tracking server configuration."""
    print(f"🔧 Setting up MLflow tracking for environment: {environment}")
    
    # Create MLflow directories
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    
    artifacts_dir = Path("mlflow_artifacts") 
    artifacts_dir.mkdir(exist_ok=True)
    
    # Set MLflow tracking URI (local for now, will be updated for remote later)
    tracking_uri = f"file://{Path.cwd()}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✅ MLflow tracking URI set to: {tracking_uri}")
    
    # Set default artifact location
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = str(artifacts_dir.absolute())
    
    return tracking_uri

def create_experiments():
    """Create MLflow experiments for the project."""
    print("🧪 Creating MLflow experiments...")
    
    experiments = [
        {
            "name": "drug_repurposing_baseline",
            "description": "Baseline experiments with pre-trained DrugBAN model"
        },
        {
            "name": "drug_repurposing_training", 
            "description": "Model training and fine-tuning experiments"
        },
        {
            "name": "drug_repurposing_evaluation",
            "description": "Model evaluation and comparison experiments"
        },
        {
            "name": "drug_repurposing_production",
            "description": "Production model deployments and A/B tests"
        }
    ]
    
    created_experiments = []
    
    for exp_config in experiments:
        try:
            # Check if experiment already exists
            experiment = mlflow.get_experiment_by_name(exp_config["name"])
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=exp_config["name"],
                    artifact_location=f"mlflow_artifacts/{exp_config['name']}"
                )
                print(f"✅ Created experiment: {exp_config['name']} (ID: {experiment_id})")
                created_experiments.append({
                    "id": experiment_id,
                    "name": exp_config["name"]
                })
            else:
                print(f"📋 Experiment already exists: {exp_config['name']} (ID: {experiment.experiment_id})")
                created_experiments.append({
                    "id": experiment.experiment_id,
                    "name": exp_config["name"]
                })
        except Exception as e:
            print(f"❌ Failed to create experiment {exp_config['name']}: {e}")
    
    return created_experiments

def test_mlflow_connection():
    """Test MLflow tracking connection."""
    print("🔍 Testing MLflow connection...")
    
    try:
        # List existing experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Test creating a simple run
        with mlflow.start_run(experiment_id=experiments[0].experiment_id, run_name="connection_test") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.99)
            print(f"✅ Test run created: {run.info.run_id}")
        
        return True
    except Exception as e:
        print(f"❌ MLflow connection test failed: {e}")
        return False

def create_mlflow_config():
    """Create MLflow configuration files."""
    print("📝 Creating MLflow configuration files...")
    
    # Create .mlflow directory
    mlflow_config_dir = Path(".mlflow")
    mlflow_config_dir.mkdir(exist_ok=True)
    
    # Create MLflow server configuration
    server_config = {
        "backend_store_uri": "sqlite:///mlflow.db",
        "default_artifact_root": "mlflow_artifacts",
        "host": "127.0.0.1",
        "port": 5000,
        "workers": 1
    }
    
    with open(mlflow_config_dir / "server_config.yaml", "w") as f:
        yaml.dump(server_config, f, default_flow_style=False)
    
    # Create environment-specific configurations
    environments = ["dev", "staging", "prod"]
    for env in environments:
        env_config = {
            "tracking_uri": f"file://{Path.cwd()}/mlruns_{env}",
            "artifact_root": f"mlflow_artifacts_{env}",
            "experiment_prefix": f"{env}_",
            "tags": {
                "environment": env,
                "project": "drug_repurposing"
            }
        }
        
        with open(mlflow_config_dir / f"{env}_config.yaml", "w") as f:
            yaml.dump(env_config, f, default_flow_style=False)
    
    print("✅ MLflow configuration files created")

def start_mlflow_server():
    """Start MLflow tracking server."""
    print("🚀 Starting MLflow tracking server...")
    
    try:
        # Check if server is already running
        import requests
        response = requests.get("http://127.0.0.1:5000/health")
        print("📋 MLflow server is already running at http://127.0.0.1:5000")
        return True
    except:
        pass
    
    try:
        # Start MLflow server in background
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "mlflow_artifacts",
            "--host", "127.0.0.1",
            "--port", "5000"
        ]
        
        print("🔄 Starting MLflow server (this may take a moment)...")
        print("💡 Server will be available at: http://127.0.0.1:5000")
        print("💡 To stop the server later, use: pkill -f 'mlflow server'")
        
        # Start server as background process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        print(f"✅ MLflow server started with PID: {process.pid}")
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Test connection
        try:
            import requests
            response = requests.get("http://127.0.0.1:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ MLflow server is responding")
                return True
        except:
            pass
        
        print("⚠️  Server started but may need more time to be ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start MLflow server: {e}")
        print("💡 You can start it manually with: mlflow server --host 127.0.0.1 --port 5000")
        return False

def main():
    """Main setup function."""
    print("🚀 MLflow Setup for MLOps Drug Repurposing Project")
    print("=" * 50)
    
    # Get environment from command line or default to dev
    environment = sys.argv[1] if len(sys.argv) > 1 else "dev"
    
    try:
        # 1. Set up tracking
        tracking_uri = setup_mlflow_tracking(environment)
        
        # 2. Create configuration files
        create_mlflow_config()
        
        # 3. Create experiments
        experiments = create_experiments()
        
        # 4. Test connection
        connection_ok = test_mlflow_connection()
        
        # 5. Start server (optional) - auto start for automated setup
        if len(sys.argv) > 2 and sys.argv[2] == "--start-server":
            start_mlflow_server()
        else:
            print("💡 To start MLflow server later, run: python mlflow_setup.py dev --start-server")
        
        print("\n" + "=" * 50)
        print("✅ MLflow setup completed successfully!")
        print(f"📊 Tracking URI: {tracking_uri}")
        print(f"🧪 Created {len(experiments)} experiments")
        print("🌐 Access MLflow UI at: http://127.0.0.1:5000")
        
        print("\n🔄 Next steps:")
        print("1. Download pre-trained DrugBAN model")
        print("2. Run baseline evaluation")
        print("3. Log results to MLflow")
        
        return 0
        
    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())