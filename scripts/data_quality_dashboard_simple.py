#!/usr/bin/env python3
"""
Simple Data Quality Dashboard for MLOps Drug Repurposing Project
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_simple_dashboard():
    """Generate a simple data quality dashboard."""
    logger.info("Generating simple data quality dashboard...")
    
    try:
        # Load validation metrics
        with open("data/validation_metrics.json", "r") as f:
            metrics = json.load(f)
        
        # Create output directory
        output_dir = Path("data/validation_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall Quality Score
        readiness_level = metrics.get('data_readiness_level', 'UNKNOWN')
        overall_quality = metrics.get('overall_quality_score', 0)
        
        axes[0, 0].bar(['Overall Quality'], [overall_quality], 
                       color='green' if overall_quality >= 90 else 'orange' if overall_quality >= 70 else 'red')
        axes[0, 0].set_title(f'Overall Data Quality Score\nReadiness Level: {readiness_level}')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_ylim(0, 100)
        
        # Add score text on bar
        axes[0, 0].text(0, overall_quality + 2, f'{overall_quality:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Dataset Quality Scores
        if 'dataset_details' in metrics:
            dataset_details = metrics['dataset_details']
            datasets = list(dataset_details.keys())
            quality_scores = [details['quality_score'] for details in dataset_details.values()]
            
            colors = ['green' if score >= 90 else 'orange' if score >= 70 else 'red' for score in quality_scores]
            bars = axes[0, 1].bar(datasets, quality_scores, color=colors)
            axes[0, 1].set_title('Quality Scores by Dataset')
            axes[0, 1].set_ylabel('Quality Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add score labels on bars
            for bar, score in zip(bars, quality_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{score:.1f}', ha='center', va='bottom')
        
        # 3. Issues Count
        if 'dataset_details' in metrics:
            issues_counts = [details['issues_count'] for details in dataset_details.values()]
            colors = ['red' if count > 0 else 'green' for count in issues_counts]
            
            bars = axes[1, 0].bar(datasets, issues_counts, color=colors)
            axes[1, 0].set_title('Validation Issues by Dataset')
            axes[1, 0].set_ylabel('Number of Issues')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for bar, count in zip(bars, issues_counts):
                if count > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   str(count), ha='center', va='bottom')
        
        # 4. Data Completeness
        if 'dataset_details' in metrics:
            completeness_scores = [details['completeness_score'] for details in dataset_details.values()]
            
            bars = axes[1, 1].bar(datasets, completeness_scores, color='skyblue')
            axes[1, 1].set_title('Data Completeness by Dataset')
            axes[1, 1].set_ylabel('Completeness Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 100)
            
            # Add score labels on bars
            for bar, score in zip(bars, completeness_scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        dashboard_path = output_dir / 'data_quality_dashboard_simple.png'
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        summary_report = f"""
# Data Quality Dashboard Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Status
- **Data Readiness Level:** {readiness_level}
- **Overall Quality Score:** {overall_quality:.1f}/100
- **Datasets Validated:** {metrics.get('total_datasets', 0)}
- **Datasets Passed:** {metrics.get('datasets_passed', 0)}

## Dataset Details
"""
        
        if 'dataset_details' in metrics:
            for dataset, details in metrics['dataset_details'].items():
                summary_report += f"""
### {dataset.replace('_', ' ').title()}
- Quality Score: {details['quality_score']:.1f}/100
- Completeness: {details['completeness_score']:.1f}%
- Issues: {details['issues_count']}
- Total Rows: {details['total_rows']:,}
"""
        
        # Save summary report
        summary_path = output_dir / 'data_quality_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        # Save timestamp for historical tracking
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_file = output_dir / f"validation_metrics_{timestamp}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Simple Data Quality Dashboard Generated!")
        print(f"📊 Dashboard Image: {dashboard_path.absolute()}")
        print(f"📄 Summary Report: {summary_path.absolute()}")
        print(f"📈 Overall Quality: {overall_quality:.1f}/100")
        print(f"🏷️  Readiness Level: {readiness_level}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(generate_simple_dashboard())