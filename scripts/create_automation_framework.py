#!/usr/bin/env python3
"""
Automation Framework for MLOps Drug Repurposing Project

This script creates a simplified automation framework for data preprocessing
and pipeline orchestration without requiring full Airflow setup.
"""

import os
import json
import subprocess
import time
from pathlib import Path
import logging
from datetime import datetime
import yaml
import schedule

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipelineOrchestrator:
    """Orchestrate data preprocessing pipeline execution."""
    
    def __init__(self):
        self.pipeline_steps = [
            {
                'name': 'data_validation',
                'script': 'scripts/validate_data.py',
                'description': 'Validate data quality and generate reports'
            },
            {
                'name': 'import_labeled_datasets',
                'script': 'scripts/import_labeled_datasets.py',
                'description': 'Import and organize labeled datasets'
            },
            {
                'name': 'generate_molecular_fingerprints',
                'script': 'scripts/generate_molecular_fingerprints.py',
                'description': 'Generate molecular fingerprints and features'
            },
            {
                'name': 'apply_data_augmentation',
                'script': 'scripts/apply_data_augmentation.py',
                'description': 'Apply data augmentation techniques'
            },
            {
                'name': 'data_quality_dashboard',
                'script': 'scripts/data_quality_dashboard_simple.py',
                'description': 'Generate data quality dashboard'
            }
        ]
        
        self.execution_logs = []
    
    def execute_step(self, step):
        """Execute a single pipeline step."""
        logger.info(f"Executing: {step['name']}")
        
        start_time = datetime.now()
        
        try:
            # Set environment and execute script
            env = os.environ.copy()
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            result = subprocess.run(
                ['python', step['script']],
                cwd=Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            step_log = {
                'step_name': step['name'],
                'script': step['script'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.execution_logs.append(step_log)
            
            if result.returncode == 0:
                logger.info(f"✅ {step['name']} completed successfully in {execution_time:.2f}s")
                return True
            else:
                logger.error(f"❌ {step['name']} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {step['name']} timed out after 10 minutes")
            step_log = {
                'step_name': step['name'],
                'script': step['script'],
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'execution_time_seconds': 600,
                'return_code': -1,
                'success': False,
                'stdout': '',
                'stderr': 'Process timed out'
            }
            self.execution_logs.append(step_log)
            return False
        
        except Exception as e:
            logger.error(f"💥 {step['name']} failed with exception: {e}")
            step_log = {
                'step_name': step['name'],
                'script': step['script'],
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'execution_time_seconds': 0,
                'return_code': -2,
                'success': False,
                'stdout': '',
                'stderr': str(e)
            }
            self.execution_logs.append(step_log)
            return False
    
    def execute_pipeline(self, steps_to_run=None):
        """Execute the complete pipeline or specific steps."""
        logger.info("🚀 Starting data preprocessing pipeline...")
        
        pipeline_start = datetime.now()
        
        if steps_to_run is None:
            steps_to_run = [step['name'] for step in self.pipeline_steps]
        
        successful_steps = 0
        total_steps = len(steps_to_run)
        
        for step in self.pipeline_steps:
            if step['name'] in steps_to_run:
                success = self.execute_step(step)
                if success:
                    successful_steps += 1
                else:
                    logger.warning(f"Pipeline step {step['name']} failed, continuing with next step...")
        
        pipeline_end = datetime.now()
        total_time = (pipeline_end - pipeline_start).total_seconds()
        
        # Create execution summary
        summary = {
            'pipeline_execution': {
                'start_time': pipeline_start.isoformat(),
                'end_time': pipeline_end.isoformat(),
                'total_execution_time_seconds': total_time,
                'steps_executed': total_steps,
                'steps_successful': successful_steps,
                'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
                'overall_success': successful_steps == total_steps
            },
            'step_details': self.execution_logs
        }
        
        # Save execution log
        log_dir = Path("logs/pipeline_executions")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_execution_{pipeline_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n🏁 Pipeline Execution Complete!")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"✅ Successful Steps: {successful_steps}/{total_steps}")
        print(f"📊 Success Rate: {summary['pipeline_execution']['success_rate']:.1%}")
        print(f"📁 Execution Log: {log_file}")
        
        return summary

class AutomatedQualityChecker:
    """Automated data quality monitoring."""
    
    def __init__(self):
        self.quality_thresholds = {
            'overall_quality_score': 80,
            'data_readiness_level': ['STAGING_READY', 'PRODUCTION_READY'],
            'datasets_passed': 3
        }
    
    def check_data_quality(self):
        """Check current data quality against thresholds."""
        logger.info("Running automated data quality checks...")
        
        try:
            # Load latest validation metrics
            with open("data/validation_metrics.json", "r") as f:
                metrics = json.load(f)
            
            quality_issues = []
            quality_score = 100
            
            # Check overall quality score
            overall_quality = metrics.get('overall_quality_score', 0)
            if overall_quality < self.quality_thresholds['overall_quality_score']:
                quality_issues.append(f"Overall quality score {overall_quality} below threshold {self.quality_thresholds['overall_quality_score']}")
                quality_score -= 20
            
            # Check data readiness level
            readiness_level = metrics.get('data_readiness_level', 'UNKNOWN')
            if readiness_level not in self.quality_thresholds['data_readiness_level']:
                quality_issues.append(f"Data readiness level '{readiness_level}' not in acceptable levels {self.quality_thresholds['data_readiness_level']}")
                quality_score -= 30
            
            # Check datasets passed
            datasets_passed = metrics.get('datasets_passed', 0)
            if datasets_passed < self.quality_thresholds['datasets_passed']:
                quality_issues.append(f"Only {datasets_passed} datasets passed, minimum required: {self.quality_thresholds['datasets_passed']}")
                quality_score -= 25
            
            quality_report = {
                'check_timestamp': datetime.now().isoformat(),
                'overall_quality_score': max(0, quality_score),
                'quality_passed': len(quality_issues) == 0,
                'issues_found': quality_issues,
                'metrics_checked': metrics,
                'thresholds_used': self.quality_thresholds
            }
            
            # Save quality report
            quality_dir = Path("logs/quality_checks")
            quality_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = quality_dir / f"quality_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            if quality_issues:
                logger.warning(f"❌ Data quality issues found: {len(quality_issues)}")
                for issue in quality_issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("✅ All data quality checks passed!")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return None

def create_automation_config():
    """Create automation configuration file."""
    logger.info("Creating automation configuration...")
    
    config = {
        'automation_settings': {
            'enable_scheduled_execution': False,
            'schedule_frequency': 'daily',
            'schedule_time': '02:00',
            'auto_quality_checks': True,
            'notification_email': None,
            'retry_failed_steps': True,
            'max_retries': 2
        },
        'pipeline_steps': {
            'data_validation': {
                'enabled': True,
                'depends_on': [],
                'timeout_minutes': 10
            },
            'import_labeled_datasets': {
                'enabled': True,
                'depends_on': ['data_validation'],
                'timeout_minutes': 5
            },
            'generate_molecular_fingerprints': {
                'enabled': True,
                'depends_on': ['import_labeled_datasets'],
                'timeout_minutes': 15
            },
            'apply_data_augmentation': {
                'enabled': True,
                'depends_on': ['generate_molecular_fingerprints'],
                'timeout_minutes': 10
            },
            'data_quality_dashboard': {
                'enabled': True,
                'depends_on': ['apply_data_augmentation'],
                'timeout_minutes': 5
            }
        },
        'quality_thresholds': {
            'overall_quality_score': 80,
            'data_readiness_level': ['STAGING_READY', 'PRODUCTION_READY'],
            'datasets_passed': 3,
            'validation_success_rate': 0.9
        },
        'monitoring': {
            'track_execution_time': True,
            'log_retention_days': 30,
            'alert_on_failure': True,
            'dashboard_auto_refresh': True
        }
    }
    
    config_file = Path("automation_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Automation configuration saved to: {config_file}")
    
    return config

def create_pipeline_runner_script():
    """Create a simple pipeline runner script."""
    logger.info("Creating pipeline runner script...")
    
    runner_script = '''#!/usr/bin/env python3
"""
Pipeline Runner for MLOps Drug Repurposing Project

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --steps validation # Run specific steps
    python run_pipeline.py --quality-check    # Run quality check only
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.create_automation_framework import DataPipelineOrchestrator, AutomatedQualityChecker

def main():
    parser = argparse.ArgumentParser(description="Run MLOps data pipeline")
    parser.add_argument('--steps', nargs='+', help='Specific steps to run')
    parser.add_argument('--quality-check', action='store_true', help='Run quality check only')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.quality_check:
        # Run quality check only
        checker = AutomatedQualityChecker()
        result = checker.check_data_quality()
        if result and result['quality_passed']:
            print("✅ Quality check passed!")
            sys.exit(0)
        else:
            print("❌ Quality check failed!")
            sys.exit(1)
    else:
        # Run pipeline
        orchestrator = DataPipelineOrchestrator()
        summary = orchestrator.execute_pipeline(args.steps)
        
        if summary['pipeline_execution']['overall_success']:
            print("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("❌ Pipeline completed with errors!")
            sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    runner_file = Path("run_pipeline.py")
    with open(runner_file, 'w') as f:
        f.write(runner_script)
    
    runner_file.chmod(0o755)
    
    logger.info(f"Pipeline runner script created: {runner_file}")
    
    return runner_file

def main():
    """Main automation framework setup function."""
    logger.info("Setting up automation framework...")
    
    try:
        # Create configuration
        config = create_automation_config()
        
        # Create pipeline runner
        runner_script = create_pipeline_runner_script()
        
        # Test the pipeline orchestrator
        orchestrator = DataPipelineOrchestrator()
        quality_checker = AutomatedQualityChecker()
        
        # Run a quality check
        quality_report = quality_checker.check_data_quality()
        
        print("\n🔧 Automation Framework Setup Complete!")
        print("=" * 50)
        print(f"⚙️  Configuration: automation_config.yaml")
        print(f"🚀 Pipeline Runner: {runner_script}")
        print()
        print("📋 Available Commands:")
        print("  python run_pipeline.py                    # Run full pipeline")
        print("  python run_pipeline.py --quality-check    # Check data quality")
        print("  python run_pipeline.py --steps validation # Run specific steps")
        print()
        print("🔍 Quality Check Results:")
        if quality_report:
            print(f"  Overall Score: {quality_report['overall_quality_score']}/100")
            print(f"  Status: {'✅ PASSED' if quality_report['quality_passed'] else '❌ FAILED'}")
            if quality_report['issues_found']:
                print("  Issues:")
                for issue in quality_report['issues_found']:
                    print(f"    - {issue}")
        else:
            print("  ⚠️  Could not run quality check (run pipeline first)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Automation framework setup failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())