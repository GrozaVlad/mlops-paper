#!/usr/bin/env python3
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
