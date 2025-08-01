#!/usr/bin/env python3
"""
Data Quality Monitoring Dashboard for MLOps Drug Repurposing Project

This script creates an interactive dashboard for monitoring data quality metrics
using Streamlit or generating static HTML reports.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityDashboard:
    """Data Quality Monitoring Dashboard."""
    
    def __init__(self, validation_reports_dir: str = "data/validation_outputs"):
        self.validation_reports_dir = Path(validation_reports_dir)
        self.validation_reports_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics = self.load_latest_metrics()
        self.historical_metrics = self.load_historical_metrics()
    
    def load_latest_metrics(self) -> Dict[str, Any]:
        """Load the latest validation metrics."""
        try:
            with open("data/validation_metrics.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No validation metrics found. Run validation first.")
            return {}
    
    def load_historical_metrics(self) -> List[Dict[str, Any]]:
        """Load historical validation metrics."""
        historical_data = []
        
        # Look for timestamped metrics files
        metrics_files = list(self.validation_reports_dir.glob("validation_metrics_*.json"))
        
        for file_path in sorted(metrics_files):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    historical_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        # If no historical data, create from current metrics
        if not historical_data and self.current_metrics:
            historical_data = [self.current_metrics]
        
        return historical_data
    
    def create_quality_score_gauge(self, score: float, title: str) -> go.Figure:
        """Create a gauge chart for quality score."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 90},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_dataset_comparison_chart(self) -> go.Figure:
        """Create a comparison chart of dataset quality scores."""
        if not self.current_metrics or 'dataset_details' not in self.current_metrics:
            return go.Figure()
        
        dataset_details = self.current_metrics['dataset_details']
        
        datasets = list(dataset_details.keys())
        quality_scores = [details['quality_score'] for details in dataset_details.values()]
        completeness_scores = [details['completeness_score'] for details in dataset_details.values()]
        validity_scores = [details['validity_score'] for details in dataset_details.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Quality Score',
            x=datasets,
            y=quality_scores,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Completeness Score',
            x=datasets,
            y=completeness_scores,
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Validity Score',
            x=datasets,
            y=validity_scores,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Dataset Quality Comparison',
            xaxis_title='Dataset',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        
        return fig\n    \n    def create_historical_trend_chart(self) -> go.Figure:\n        \"\"\"Create a historical trend chart of quality scores.\"\"\"\n        if not self.historical_metrics:\n            return go.Figure()\n        \n        # Extract timestamps and overall quality scores\n        timestamps = []\n        overall_scores = []\n        dataset_scores = {}\n        \n        for metrics in self.historical_metrics:\n            timestamp = metrics.get('validation_timestamp', datetime.now().isoformat())\n            timestamps.append(pd.to_datetime(timestamp))\n            overall_scores.append(metrics.get('overall_quality_score', 0))\n            \n            # Extract individual dataset scores\n            if 'dataset_details' in metrics:\n                for dataset, details in metrics['dataset_details'].items():\n                    if dataset not in dataset_scores:\n                        dataset_scores[dataset] = []\n                    dataset_scores[dataset].append(details.get('quality_score', 0))\n        \n        fig = go.Figure()\n        \n        # Add overall trend\n        fig.add_trace(go.Scatter(\n            x=timestamps,\n            y=overall_scores,\n            mode='lines+markers',\n            name='Overall Quality',\n            line=dict(width=3, color='blue')\n        ))\n        \n        # Add individual dataset trends\n        colors = ['red', 'green', 'orange', 'purple', 'brown']\n        for i, (dataset, scores) in enumerate(dataset_scores.items()):\n            if len(scores) == len(timestamps):  # Ensure data consistency\n                fig.add_trace(go.Scatter(\n                    x=timestamps,\n                    y=scores,\n                    mode='lines+markers',\n                    name=dataset.replace('_', ' ').title(),\n                    line=dict(color=colors[i % len(colors)])\n                ))\n        \n        fig.update_layout(\n            title='Data Quality Trends Over Time',\n            xaxis_title='Date',\n            yaxis_title='Quality Score',\n            height=400,\n            hovermode='x unified'\n        )\n        \n        return fig\n    \n    def create_issues_breakdown_chart(self) -> go.Figure:\n        \"\"\"Create a breakdown chart of validation issues.\"\"\"\n        if not self.current_metrics or 'dataset_details' not in self.current_metrics:\n            return go.Figure()\n        \n        dataset_details = self.current_metrics['dataset_details']\n        \n        datasets = list(dataset_details.keys())\n        issues_counts = [details['issues_count'] for details in dataset_details.values()]\n        \n        colors = ['red' if count > 0 else 'green' for count in issues_counts]\n        \n        fig = go.Figure(go.Bar(\n            x=datasets,\n            y=issues_counts,\n            marker_color=colors,\n            text=issues_counts,\n            textposition='auto'\n        ))\n        \n        fig.update_layout(\n            title='Validation Issues by Dataset',\n            xaxis_title='Dataset',\n            yaxis_title='Number of Issues',\n            height=300\n        )\n        \n        return fig\n    \n    def create_data_readiness_summary(self) -> Dict[str, Any]:\n        \"\"\"Create a summary of data readiness status.\"\"\"\n        if not self.current_metrics:\n            return {}\n        \n        readiness_level = self.current_metrics.get('data_readiness_level', 'UNKNOWN')\n        overall_quality = self.current_metrics.get('overall_quality_score', 0)\n        total_datasets = self.current_metrics.get('total_datasets', 0)\n        datasets_passed = self.current_metrics.get('datasets_passed', 0)\n        \n        readiness_colors = {\n            'PRODUCTION_READY': 'green',\n            'STAGING_READY': 'blue',\n            'DEVELOPMENT_READY': 'orange',\n            'NEEDS_IMPROVEMENT': 'red',\n            'UNKNOWN': 'gray'\n        }\n        \n        return {\n            'readiness_level': readiness_level,\n            'readiness_color': readiness_colors.get(readiness_level, 'gray'),\n            'overall_quality': overall_quality,\n            'datasets_passed': datasets_passed,\n            'total_datasets': total_datasets,\n            'pass_rate': (datasets_passed / total_datasets * 100) if total_datasets > 0 else 0\n        }\n    \n    def create_molecular_properties_chart(self) -> go.Figure:\n        \"\"\"Create charts for molecular properties analysis.\"\"\"\n        # This would require loading the validation report with molecular properties\n        try:\n            with open(\"data/validation_report.json\", \"r\") as f:\n                report = json.load(f)\n            \n            # Look for drug metadata validation results\n            for result in report.get('validation_results', []):\n                if result.get('dataset_name') == 'drug_metadata' and 'molecular_properties_stats' in result:\n                    props = result['molecular_properties_stats']\n                    \n                    # Create subplots for molecular properties\n                    fig = make_subplots(\n                        rows=2, cols=2,\n                        subplot_titles=('Molecular Weight Distribution', 'LogP Distribution', \n                                       'Drug-likeness Violations', 'Binding Properties'),\n                        specs=[[{\"type\": \"histogram\"}, {\"type\": \"histogram\"}],\n                               [{\"type\": \"bar\"}, {\"type\": \"bar\"}]]\n                    )\n                    \n                    # Molecular weight histogram (simulated data based on stats)\n                    if 'molecular_weight' in props:\n                        mw_stats = props['molecular_weight']\n                        # Generate sample data for visualization\n                        mw_data = np.random.normal(mw_stats['mean'], mw_stats['std'], 1000)\n                        mw_data = np.clip(mw_data, mw_stats['min'], mw_stats['max'])\n                        \n                        fig.add_trace(\n                            go.Histogram(x=mw_data, name=\"Molecular Weight\"),\n                            row=1, col=1\n                        )\n                    \n                    # LogP histogram\n                    if 'logp' in props:\n                        logp_stats = props['logp']\n                        logp_data = np.random.normal(logp_stats['mean'], logp_stats['std'], 1000)\n                        \n                        fig.add_trace(\n                            go.Histogram(x=logp_data, name=\"LogP\"),\n                            row=1, col=2\n                        )\n                    \n                    # Drug-likeness violations\n                    if 'drug_likeness' in props:\n                        violations = props['drug_likeness']['lipinski_violations']\n                        total_drugs = 1000  # Estimate\n                        \n                        fig.add_trace(\n                            go.Bar(x=['Compliant', 'Violations'], \n                                  y=[total_drugs - violations, violations],\n                                  name=\"Lipinski Rule\"),\n                            row=2, col=1\n                        )\n                    \n                    fig.update_layout(\n                        title='Molecular Properties Analysis',\n                        height=600,\n                        showlegend=False\n                    )\n                    \n                    return fig\n                    \n        except Exception as e:\n            logger.warning(f\"Could not create molecular properties chart: {e}\")\n        \n        return go.Figure()\n    \n    def generate_html_dashboard(self, output_file: str = \"data_quality_dashboard.html\") -> str:\n        \"\"\"Generate a complete HTML dashboard.\"\"\"\n        logger.info(\"Generating data quality dashboard...\")\n        \n        # Create all charts\n        readiness_summary = self.create_data_readiness_summary()\n        \n        # Overall quality gauge\n        overall_quality = readiness_summary.get('overall_quality', 0)\n        quality_gauge = self.create_quality_score_gauge(overall_quality, \"Overall Data Quality\")\n        \n        # Dataset comparison\n        dataset_comparison = self.create_dataset_comparison_chart()\n        \n        # Historical trends\n        historical_trends = self.create_historical_trend_chart()\n        \n        # Issues breakdown\n        issues_breakdown = self.create_issues_breakdown_chart()\n        \n        # Molecular properties\n        molecular_props = self.create_molecular_properties_chart()\n        \n        # Create HTML content\n        html_content = f\"\"\"\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>Data Quality Dashboard - MLOps Drug Repurposing</title>\n            <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}\n                .header {{ text-align: center; background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}\n                .summary-card {{ background-color: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-block; min-width: 200px; }}\n                .chart-container {{ background-color: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}\n                .readiness-badge {{ padding: 10px 20px; border-radius: 20px; color: white; font-weight: bold; display: inline-block; }}\n                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}\n            </style>\n        </head>\n        <body>\n            <div class=\"header\">\n                <h1>Data Quality Dashboard</h1>\n                <p>MLOps Drug Repurposing Project</p>\n                <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n            </div>\n            \n            <div style=\"text-align: center; margin: 20px 0;\">\n                <div class=\"summary-card\">\n                    <h3>Data Readiness Level</h3>\n                    <div class=\"readiness-badge\" style=\"background-color: {readiness_summary.get('readiness_color', 'gray')}\">\n                        {readiness_summary.get('readiness_level', 'UNKNOWN')}\n                    </div>\n                </div>\n                \n                <div class=\"summary-card\">\n                    <h3>Overall Quality Score</h3>\n                    <h2 style=\"color: {readiness_summary.get('readiness_color', 'gray')}\">\n                        {overall_quality:.1f}/100\n                    </h2>\n                </div>\n                \n                <div class=\"summary-card\">\n                    <h3>Dataset Pass Rate</h3>\n                    <h2>{readiness_summary.get('pass_rate', 0):.1f}%</h2>\n                    <p>({readiness_summary.get('datasets_passed', 0)}/{readiness_summary.get('total_datasets', 0)} datasets)</p>\n                </div>\n            </div>\n            \n            <div class=\"chart-container\">\n                <div id=\"quality-gauge\"></div>\n            </div>\n            \n            <div class=\"grid\">\n                <div class=\"chart-container\">\n                    <div id=\"dataset-comparison\"></div>\n                </div>\n                \n                <div class=\"chart-container\">\n                    <div id=\"issues-breakdown\"></div>\n                </div>\n            </div>\n            \n            <div class=\"chart-container\">\n                <div id=\"historical-trends\"></div>\n            </div>\n            \n            <div class=\"chart-container\">\n                <div id=\"molecular-properties\"></div>\n            </div>\n            \n            <script>\n                Plotly.newPlot('quality-gauge', {quality_gauge.to_json()});\n                Plotly.newPlot('dataset-comparison', {dataset_comparison.to_json()});\n                Plotly.newPlot('issues-breakdown', {issues_breakdown.to_json()});\n                Plotly.newPlot('historical-trends', {historical_trends.to_json()});\n                Plotly.newPlot('molecular-properties', {molecular_props.to_json()});\n            </script>\n        </body>\n        </html>\n        \"\"\"\n        \n        # Write HTML file\n        output_path = Path(output_file)\n        with open(output_path, 'w') as f:\n            f.write(html_content)\n        \n        logger.info(f\"Dashboard saved to: {output_path.absolute()}\")\n        return str(output_path.absolute())\n    \n    def save_metrics_snapshot(self) -> None:\n        \"\"\"Save current metrics as a timestamped snapshot for historical tracking.\"\"\"\n        if not self.current_metrics:\n            return\n        \n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        snapshot_file = self.validation_reports_dir / f\"validation_metrics_{timestamp}.json\"\n        \n        with open(snapshot_file, 'w') as f:\n            json.dump(self.current_metrics, f, indent=2)\n        \n        logger.info(f\"Metrics snapshot saved to: {snapshot_file}\")\n\ndef main():\n    \"\"\"Main function to generate the dashboard.\"\"\"\n    logger.info(\"Starting data quality dashboard generation...\")\n    \n    try:\n        # Create dashboard\n        dashboard = DataQualityDashboard()\n        \n        # Save metrics snapshot for historical tracking\n        dashboard.save_metrics_snapshot()\n        \n        # Generate HTML dashboard\n        dashboard_path = dashboard.generate_html_dashboard(\"data/validation_outputs/data_quality_dashboard.html\")\n        \n        print(f\"\\n✅ Data Quality Dashboard Generated Successfully!\")\n        print(f\"📊 Dashboard Location: {dashboard_path}\")\n        print(f\"🌐 Open in browser: file://{dashboard_path}\")\n        \n        # Print summary\n        readiness = dashboard.create_data_readiness_summary()\n        if readiness:\n            print(f\"\\n📈 Summary:\")\n            print(f\"   Readiness Level: {readiness.get('readiness_level', 'UNKNOWN')}\")\n            print(f\"   Overall Quality: {readiness.get('overall_quality', 0):.1f}/100\")\n            print(f\"   Pass Rate: {readiness.get('pass_rate', 0):.1f}%\")\n        \n        return 0\n        \n    except Exception as e:\n        logger.error(f\"Dashboard generation failed: {e}\")\n        import traceback\n        logger.error(f\"Full traceback: {traceback.format_exc()}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())