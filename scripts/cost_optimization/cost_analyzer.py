#!/usr/bin/env python3
"""
Infrastructure Cost Analyzer and Optimizer

Analyzes cloud infrastructure costs and provides optimization recommendations
for the MLOps pipeline.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Cloud resource types"""
    EC2_INSTANCE = "ec2_instance"
    EKS_CLUSTER = "eks_cluster"
    RDS_INSTANCE = "rds_instance"
    S3_BUCKET = "s3_bucket"
    EBS_VOLUME = "ebs_volume"
    LOAD_BALANCER = "load_balancer"
    NAT_GATEWAY = "nat_gateway"
    DATA_TRANSFER = "data_transfer"
    LAMBDA_FUNCTION = "lambda_function"
    ELASTICACHE = "elasticache"


class OptimizationType(Enum):
    """Types of cost optimizations"""
    RIGHT_SIZING = "right_sizing"
    RESERVED_INSTANCES = "reserved_instances"
    SPOT_INSTANCES = "spot_instances"
    STORAGE_CLASS = "storage_class"
    LIFECYCLE_POLICY = "lifecycle_policy"
    IDLE_RESOURCE = "idle_resource"
    SCHEDULING = "scheduling"
    ARCHITECTURE = "architecture"
    CACHING = "caching"
    DATA_TRANSFER = "data_transfer"


@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    resource_id: str
    resource_type: ResourceType
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    network_in: Optional[float] = None
    network_out: Optional[float] = None
    storage_used: Optional[float] = None
    request_count: Optional[int] = None
    active_hours: Optional[float] = None
    idle_hours: Optional[float] = None


@dataclass
class CostBreakdown:
    """Cost breakdown for a resource"""
    resource_id: str
    resource_type: ResourceType
    daily_cost: float
    monthly_cost: float
    yearly_cost: float
    cost_components: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    resource_id: str
    resource_type: ResourceType
    optimization_type: OptimizationType
    current_cost: float
    optimized_cost: float
    savings: float
    savings_percentage: float
    description: str
    implementation_steps: List[str]
    risk_level: str  # low, medium, high
    effort_level: str  # low, medium, high
    priority: int  # 1-10, higher is more important


class CostAnalyzer:
    """Infrastructure cost analyzer and optimizer"""
    
    def __init__(self, 
                 aws_profile: Optional[str] = None,
                 region: str = "us-east-1",
                 config_path: str = "configs/cost_optimization_config.yaml"):
        """
        Initialize cost analyzer
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
            config_path: Path to configuration file
        """
        self.region = region
        self.config = self._load_config(config_path)
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.ec2_client = session.client('ec2', region_name=region)
        self.cloudwatch_client = session.client('cloudwatch', region_name=region)
        self.ce_client = session.client('ce', region_name=region)  # Cost Explorer
        self.s3_client = session.client('s3', region_name=region)
        self.rds_client = session.client('rds', region_name=region)
        self.eks_client = session.client('eks', region_name=region)
        
        # Cost data cache
        self.cost_cache = {}
        self.usage_cache = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                'thresholds': {
                    'cpu_utilization_low': 20,
                    'cpu_utilization_high': 80,
                    'memory_utilization_low': 30,
                    'memory_utilization_high': 85,
                    'idle_hours_threshold': 12
                },
                'optimization': {
                    'enable_spot_instances': True,
                    'enable_reserved_instances': True,
                    'enable_scheduling': True
                }
            }
    
    def analyze_costs(self, 
                     start_date: datetime,
                     end_date: datetime,
                     granularity: str = "DAILY") -> Dict[str, Any]:
        """
        Analyze infrastructure costs for a given period
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            granularity: Cost granularity (DAILY, MONTHLY)
            
        Returns:
            Cost analysis results
        """
        logger.info(f"Analyzing costs from {start_date} to {end_date}")
        
        # Get cost and usage data
        cost_data = self._get_cost_and_usage(start_date, end_date, granularity)
        
        # Get resource inventory
        resources = self._get_resource_inventory()
        
        # Analyze resource usage
        usage_data = self._analyze_resource_usage(resources, start_date, end_date)
        
        # Generate cost breakdowns
        cost_breakdowns = self._generate_cost_breakdowns(cost_data, resources)
        
        # Identify optimization opportunities
        recommendations = self._identify_optimizations(cost_breakdowns, usage_data)
        
        # Generate cost forecast
        forecast = self._generate_cost_forecast(cost_data)
        
        # Compile analysis results
        analysis = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_cost': sum(cb.monthly_cost for cb in cost_breakdowns),
            'cost_by_service': self._aggregate_by_service(cost_breakdowns),
            'cost_by_resource_type': self._aggregate_by_resource_type(cost_breakdowns),
            'top_expensive_resources': self._get_top_expensive_resources(cost_breakdowns, 10),
            'recommendations': recommendations,
            'potential_savings': sum(r.savings for r in recommendations),
            'forecast': forecast,
            'resource_count': len(resources),
            'analyzed_at': datetime.utcnow().isoformat()
        }
        
        return analysis
    
    def _get_cost_and_usage(self, 
                           start_date: datetime,
                           end_date: datetime,
                           granularity: str) -> Dict[str, Any]:
        """Get cost and usage data from AWS Cost Explorer"""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity=granularity,
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
                ]
            )
            
            return response['ResultsByTime']
            
        except ClientError as e:
            logger.error(f"Error getting cost data: {str(e)}")
            return {}
    
    def _get_resource_inventory(self) -> List[Dict[str, Any]]:
        """Get inventory of all cloud resources"""
        resources = []
        
        # Get EC2 instances
        try:
            ec2_response = self.ec2_client.describe_instances()
            for reservation in ec2_response['Reservations']:
                for instance in reservation['Instances']:
                    if instance['State']['Name'] != 'terminated':
                        resources.append({
                            'id': instance['InstanceId'],
                            'type': ResourceType.EC2_INSTANCE,
                            'instance_type': instance['InstanceType'],
                            'state': instance['State']['Name'],
                            'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                        })
        except ClientError as e:
            logger.error(f"Error getting EC2 instances: {str(e)}")
        
        # Get RDS instances
        try:
            rds_response = self.rds_client.describe_db_instances()
            for db in rds_response['DBInstances']:
                resources.append({
                    'id': db['DBInstanceIdentifier'],
                    'type': ResourceType.RDS_INSTANCE,
                    'instance_class': db['DBInstanceClass'],
                    'engine': db['Engine'],
                    'allocated_storage': db['AllocatedStorage'],
                    'tags': {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
                })
        except ClientError as e:
            logger.error(f"Error getting RDS instances: {str(e)}")
        
        # Get S3 buckets
        try:
            s3_response = self.s3_client.list_buckets()
            for bucket in s3_response['Buckets']:
                resources.append({
                    'id': bucket['Name'],
                    'type': ResourceType.S3_BUCKET,
                    'creation_date': bucket['CreationDate']
                })
        except ClientError as e:
            logger.error(f"Error getting S3 buckets: {str(e)}")
        
        # Get EKS clusters
        try:
            eks_response = self.eks_client.list_clusters()
            for cluster_name in eks_response['clusters']:
                resources.append({
                    'id': cluster_name,
                    'type': ResourceType.EKS_CLUSTER
                })
        except ClientError as e:
            logger.error(f"Error getting EKS clusters: {str(e)}")
        
        return resources
    
    def _analyze_resource_usage(self, 
                               resources: List[Dict[str, Any]],
                               start_date: datetime,
                               end_date: datetime) -> Dict[str, ResourceUsage]:
        """Analyze usage patterns for resources"""
        usage_data = {}
        
        for resource in resources:
            if resource['type'] == ResourceType.EC2_INSTANCE:
                usage = self._analyze_ec2_usage(resource['id'], start_date, end_date)
                if usage:
                    usage_data[resource['id']] = usage
            elif resource['type'] == ResourceType.RDS_INSTANCE:
                usage = self._analyze_rds_usage(resource['id'], start_date, end_date)
                if usage:
                    usage_data[resource['id']] = usage
        
        return usage_data
    
    def _analyze_ec2_usage(self, 
                          instance_id: str,
                          start_date: datetime,
                          end_date: datetime) -> Optional[ResourceUsage]:
        """Analyze EC2 instance usage"""
        try:
            # Get CPU utilization
            cpu_response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_date,
                EndTime=end_date,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            
            cpu_data = cpu_response['Datapoints']
            cpu_avg = np.mean([d['Average'] for d in cpu_data]) if cpu_data else 0
            
            # Get network metrics
            network_in_response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='NetworkIn',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_date,
                EndTime=end_date,
                Period=3600,
                Statistics=['Sum']
            )
            
            network_in = sum(d['Sum'] for d in network_in_response['Datapoints'])
            
            # Calculate idle hours (CPU < threshold)
            idle_threshold = self.config['thresholds']['cpu_utilization_low']
            idle_hours = sum(1 for d in cpu_data if d['Average'] < idle_threshold)
            active_hours = len(cpu_data) - idle_hours
            
            return ResourceUsage(
                resource_id=instance_id,
                resource_type=ResourceType.EC2_INSTANCE,
                cpu_utilization=cpu_avg,
                network_in=network_in,
                active_hours=active_hours,
                idle_hours=idle_hours
            )
            
        except ClientError as e:
            logger.error(f"Error analyzing EC2 usage for {instance_id}: {str(e)}")
            return None
    
    def _analyze_rds_usage(self,
                          db_instance_id: str,
                          start_date: datetime,
                          end_date: datetime) -> Optional[ResourceUsage]:
        """Analyze RDS instance usage"""
        try:
            # Get CPU utilization
            cpu_response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_date,
                EndTime=end_date,
                Period=3600,
                Statistics=['Average']
            )
            
            cpu_data = cpu_response['Datapoints']
            cpu_avg = np.mean([d['Average'] for d in cpu_data]) if cpu_data else 0
            
            # Get connection count
            conn_response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName='DatabaseConnections',
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_instance_id}],
                StartTime=start_date,
                EndTime=end_date,
                Period=3600,
                Statistics=['Average']
            )
            
            conn_avg = np.mean([d['Average'] for d in conn_response['Datapoints']) if conn_response['Datapoints'] else 0
            
            return ResourceUsage(
                resource_id=db_instance_id,
                resource_type=ResourceType.RDS_INSTANCE,
                cpu_utilization=cpu_avg,
                request_count=int(conn_avg)
            )
            
        except ClientError as e:
            logger.error(f"Error analyzing RDS usage for {db_instance_id}: {str(e)}")
            return None
    
    def _generate_cost_breakdowns(self,
                                 cost_data: List[Dict[str, Any]],
                                 resources: List[Dict[str, Any]]) -> List[CostBreakdown]:
        """Generate cost breakdowns for resources"""
        breakdowns = []
        
        # Process cost data
        for period in cost_data:
            for group in period.get('Groups', []):
                service = group['Keys'][0]
                usage_type = group['Keys'][1]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                
                # Map to resources
                for resource in resources:
                    if self._matches_resource(resource, service, usage_type):
                        breakdown = CostBreakdown(
                            resource_id=resource['id'],
                            resource_type=resource['type'],
                            daily_cost=cost,
                            monthly_cost=cost * 30,
                            yearly_cost=cost * 365,
                            cost_components={usage_type: cost},
                            tags=resource.get('tags', {})
                        )
                        breakdowns.append(breakdown)
        
        return breakdowns
    
    def _matches_resource(self, 
                         resource: Dict[str, Any],
                         service: str,
                         usage_type: str) -> bool:
        """Check if cost data matches a resource"""
        # Simple matching logic - can be enhanced
        if resource['type'] == ResourceType.EC2_INSTANCE and 'EC2' in service:
            return True
        elif resource['type'] == ResourceType.RDS_INSTANCE and 'RDS' in service:
            return True
        elif resource['type'] == ResourceType.S3_BUCKET and 'S3' in service:
            return True
        elif resource['type'] == ResourceType.EKS_CLUSTER and 'EKS' in service:
            return True
        return False
    
    def _identify_optimizations(self,
                               cost_breakdowns: List[CostBreakdown],
                               usage_data: Dict[str, ResourceUsage]) -> List[OptimizationRecommendation]:
        """Identify cost optimization opportunities"""
        recommendations = []
        
        for breakdown in cost_breakdowns:
            usage = usage_data.get(breakdown.resource_id)
            
            if not usage:
                continue
            
            # Check for right-sizing opportunities
            if usage.cpu_utilization and usage.cpu_utilization < self.config['thresholds']['cpu_utilization_low']:
                rec = self._create_rightsizing_recommendation(breakdown, usage)
                if rec:
                    recommendations.append(rec)
            
            # Check for idle resources
            if usage.idle_hours and usage.idle_hours > self.config['thresholds']['idle_hours_threshold']:
                rec = self._create_idle_resource_recommendation(breakdown, usage)
                if rec:
                    recommendations.append(rec)
            
            # Check for reserved instance opportunities
            if breakdown.resource_type in [ResourceType.EC2_INSTANCE, ResourceType.RDS_INSTANCE]:
                rec = self._create_reserved_instance_recommendation(breakdown)
                if rec:
                    recommendations.append(rec)
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.savings, reverse=True)
        
        # Assign priorities
        for i, rec in enumerate(recommendations):
            rec.priority = min(10, 10 - i // 3)
        
        return recommendations
    
    def _create_rightsizing_recommendation(self,
                                          breakdown: CostBreakdown,
                                          usage: ResourceUsage) -> Optional[OptimizationRecommendation]:
        """Create right-sizing recommendation"""
        if breakdown.resource_type != ResourceType.EC2_INSTANCE:
            return None
        
        # Estimate savings from downsizing (simplified)
        current_cost = breakdown.monthly_cost
        optimized_cost = current_cost * 0.6  # Assume 40% savings from right-sizing
        savings = current_cost - optimized_cost
        
        return OptimizationRecommendation(
            resource_id=breakdown.resource_id,
            resource_type=breakdown.resource_type,
            optimization_type=OptimizationType.RIGHT_SIZING,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings=savings,
            savings_percentage=(savings / current_cost) * 100,
            description=f"Instance has low CPU utilization ({usage.cpu_utilization:.1f}%). Consider downsizing.",
            implementation_steps=[
                "Review instance workload patterns",
                "Identify appropriate smaller instance type",
                "Test performance with smaller instance",
                "Schedule maintenance window for resize",
                "Monitor performance after resize"
            ],
            risk_level="medium",
            effort_level="medium",
            priority=0
        )
    
    def _create_idle_resource_recommendation(self,
                                           breakdown: CostBreakdown,
                                           usage: ResourceUsage) -> Optional[OptimizationRecommendation]:
        """Create idle resource recommendation"""
        idle_percentage = (usage.idle_hours / (usage.idle_hours + usage.active_hours)) * 100
        
        if idle_percentage < 50:
            return None
        
        # Calculate savings from scheduling
        current_cost = breakdown.monthly_cost
        optimized_cost = current_cost * (usage.active_hours / (usage.idle_hours + usage.active_hours))
        savings = current_cost - optimized_cost
        
        return OptimizationRecommendation(
            resource_id=breakdown.resource_id,
            resource_type=breakdown.resource_type,
            optimization_type=OptimizationType.SCHEDULING,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings=savings,
            savings_percentage=(savings / current_cost) * 100,
            description=f"Resource is idle {idle_percentage:.1f}% of the time. Consider scheduling or termination.",
            implementation_steps=[
                "Analyze usage patterns",
                "Implement auto-scaling or scheduling",
                "Configure start/stop automation",
                "Set up monitoring alerts",
                "Document scheduling policy"
            ],
            risk_level="low",
            effort_level="low",
            priority=0
        )
    
    def _create_reserved_instance_recommendation(self,
                                               breakdown: CostBreakdown) -> Optional[OptimizationRecommendation]:
        """Create reserved instance recommendation"""
        # Estimate RI savings (typically 30-70%)
        current_cost = breakdown.yearly_cost
        optimized_cost = current_cost * 0.6  # Assume 40% savings with RI
        savings = current_cost - optimized_cost
        
        return OptimizationRecommendation(
            resource_id=breakdown.resource_id,
            resource_type=breakdown.resource_type,
            optimization_type=OptimizationType.RESERVED_INSTANCES,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings=savings,
            savings_percentage=(savings / current_cost) * 100,
            description="Consider purchasing Reserved Instances for long-running resources.",
            implementation_steps=[
                "Analyze historical usage (12+ months)",
                "Determine commitment term (1 or 3 years)",
                "Choose payment option (All Upfront, Partial, No Upfront)",
                "Purchase Reserved Instances",
                "Monitor utilization and coverage"
            ],
            risk_level="low",
            effort_level="low",
            priority=0
        )
    
    def _aggregate_by_service(self, breakdowns: List[CostBreakdown]) -> Dict[str, float]:
        """Aggregate costs by service"""
        costs_by_service = defaultdict(float)
        
        service_mapping = {
            ResourceType.EC2_INSTANCE: "EC2",
            ResourceType.RDS_INSTANCE: "RDS",
            ResourceType.S3_BUCKET: "S3",
            ResourceType.EKS_CLUSTER: "EKS",
            ResourceType.EBS_VOLUME: "EBS",
            ResourceType.LOAD_BALANCER: "ELB",
            ResourceType.NAT_GATEWAY: "VPC"
        }
        
        for breakdown in breakdowns:
            service = service_mapping.get(breakdown.resource_type, "Other")
            costs_by_service[service] += breakdown.monthly_cost
        
        return dict(costs_by_service)
    
    def _aggregate_by_resource_type(self, breakdowns: List[CostBreakdown]) -> Dict[str, float]:
        """Aggregate costs by resource type"""
        costs_by_type = defaultdict(float)
        
        for breakdown in breakdowns:
            costs_by_type[breakdown.resource_type.value] += breakdown.monthly_cost
        
        return dict(costs_by_type)
    
    def _get_top_expensive_resources(self, 
                                   breakdowns: List[CostBreakdown],
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Get top expensive resources"""
        sorted_breakdowns = sorted(breakdowns, key=lambda x: x.monthly_cost, reverse=True)
        
        return [
            {
                'resource_id': b.resource_id,
                'resource_type': b.resource_type.value,
                'monthly_cost': b.monthly_cost,
                'tags': b.tags
            }
            for b in sorted_breakdowns[:limit]
        ]
    
    def _generate_cost_forecast(self, cost_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cost forecast based on historical data"""
        if not cost_data:
            return {}
        
        # Extract daily costs
        daily_costs = []
        for period in cost_data:
            total_cost = sum(
                float(group['Metrics']['UnblendedCost']['Amount'])
                for group in period.get('Groups', [])
            )
            daily_costs.append(total_cost)
        
        # Simple forecast using moving average
        if len(daily_costs) >= 7:
            weekly_avg = np.mean(daily_costs[-7:])
            monthly_forecast = weekly_avg * 30
            yearly_forecast = weekly_avg * 365
            
            # Calculate trend
            if len(daily_costs) >= 14:
                prev_week_avg = np.mean(daily_costs[-14:-7])
                trend = ((weekly_avg - prev_week_avg) / prev_week_avg) * 100
            else:
                trend = 0
            
            return {
                'daily_average': weekly_avg,
                'monthly_forecast': monthly_forecast,
                'yearly_forecast': yearly_forecast,
                'trend_percentage': trend,
                'confidence': 'medium'
            }
        
        return {}
    
    def generate_cost_report(self, 
                           analysis: Dict[str, Any],
                           output_path: str = "cost_analysis_report.html") -> None:
        """Generate HTML cost analysis report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Infrastructure Cost Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .savings {{ color: #28a745; font-weight: bold; }}
                .recommendation {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .high-priority {{ border-left: 5px solid #dc3545; }}
                .medium-priority {{ border-left: 5px solid #ffc107; }}
                .low-priority {{ border-left: 5px solid #28a745; }}
            </style>
        </head>
        <body>
            <h1>Infrastructure Cost Analysis Report</h1>
            <p>Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <h2>Executive Summary</h2>
            <div class="metric">
                <h3>Total Monthly Cost: ${analysis['total_cost']:,.2f}</h3>
                <p>Potential Monthly Savings: <span class="savings">${analysis['potential_savings']:,.2f}</span></p>
                <p>Number of Resources Analyzed: {analysis['resource_count']}</p>
            </div>
            
            <h2>Cost Breakdown by Service</h2>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Monthly Cost</th>
                    <th>Percentage</th>
                </tr>
        """
        
        # Add service breakdown
        total = analysis['total_cost']
        for service, cost in analysis['cost_by_service'].items():
            percentage = (cost / total) * 100 if total > 0 else 0
            html_content += f"""
                <tr>
                    <td>{service}</td>
                    <td>${cost:,.2f}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Top 10 Most Expensive Resources</h2>
            <table>
                <tr>
                    <th>Resource ID</th>
                    <th>Type</th>
                    <th>Monthly Cost</th>
                </tr>
        """
        
        # Add top resources
        for resource in analysis['top_expensive_resources']:
            html_content += f"""
                <tr>
                    <td>{resource['resource_id']}</td>
                    <td>{resource['resource_type']}</td>
                    <td>${resource['monthly_cost']:,.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Optimization Recommendations</h2>
        """
        
        # Add recommendations
        for rec in analysis['recommendations'][:10]:  # Top 10 recommendations
            priority_class = 'high-priority' if rec.priority >= 8 else 'medium-priority' if rec.priority >= 5 else 'low-priority'
            html_content += f"""
            <div class="recommendation {priority_class}">
                <h3>{rec.optimization_type.value.replace('_', ' ').title()}: {rec.resource_id}</h3>
                <p>{rec.description}</p>
                <p>Potential Savings: <span class="savings">${rec.savings:,.2f}/month ({rec.savings_percentage:.1f}%)</span></p>
                <p>Risk Level: {rec.risk_level} | Effort Level: {rec.effort_level}</p>
                <h4>Implementation Steps:</h4>
                <ol>
        """
            
            for step in rec.implementation_steps:
                html_content += f"<li>{step}</li>"
            
            html_content += """
                </ol>
            </div>
            """
        
        # Add forecast
        if 'forecast' in analysis and analysis['forecast']:
            forecast = analysis['forecast']
            trend_direction = "↑" if forecast['trend_percentage'] > 0 else "↓" if forecast['trend_percentage'] < 0 else "→"
            html_content += f"""
            <h2>Cost Forecast</h2>
            <div class="metric">
                <p>Daily Average: ${forecast['daily_average']:,.2f}</p>
                <p>Monthly Forecast: ${forecast['monthly_forecast']:,.2f}</p>
                <p>Yearly Forecast: ${forecast['yearly_forecast']:,.2f}</p>
                <p>Trend: {trend_direction} {abs(forecast['trend_percentage']):.1f}%</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Cost report generated: {output_path}")
    
    def export_recommendations_csv(self, 
                                 recommendations: List[OptimizationRecommendation],
                                 output_path: str = "cost_recommendations.csv") -> None:
        """Export recommendations to CSV"""
        data = []
        
        for rec in recommendations:
            data.append({
                'Resource ID': rec.resource_id,
                'Resource Type': rec.resource_type.value,
                'Optimization Type': rec.optimization_type.value,
                'Current Cost': rec.current_cost,
                'Optimized Cost': rec.optimized_cost,
                'Monthly Savings': rec.savings,
                'Savings %': rec.savings_percentage,
                'Risk Level': rec.risk_level,
                'Effort Level': rec.effort_level,
                'Priority': rec.priority,
                'Description': rec.description
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Recommendations exported to: {output_path}")
    
    def visualize_costs(self, analysis: Dict[str, Any], output_dir: str = "cost_visualizations") -> None:
        """Generate cost visualization charts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Cost by Service Pie Chart
        if 'cost_by_service' in analysis:
            plt.figure(figsize=(10, 8))
            costs = list(analysis['cost_by_service'].values())
            labels = list(analysis['cost_by_service'].keys())
            
            plt.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('Monthly Cost Distribution by Service')
            plt.savefig(f"{output_dir}/cost_by_service.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Top Resources Bar Chart
        if 'top_expensive_resources' in analysis:
            plt.figure(figsize=(12, 8))
            resources = analysis['top_expensive_resources'][:10]
            ids = [r['resource_id'][:20] + '...' if len(r['resource_id']) > 20 else r['resource_id'] for r in resources]
            costs = [r['monthly_cost'] for r in resources]
            
            plt.barh(ids, costs)
            plt.xlabel('Monthly Cost ($)')
            plt.title('Top 10 Most Expensive Resources')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_expensive_resources.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Savings Potential by Optimization Type
        if 'recommendations' in analysis:
            savings_by_type = defaultdict(float)
            for rec in analysis['recommendations']:
                savings_by_type[rec.optimization_type.value] += rec.savings
            
            plt.figure(figsize=(10, 6))
            types = list(savings_by_type.keys())
            savings = list(savings_by_type.values())
            
            plt.bar(types, savings)
            plt.xlabel('Optimization Type')
            plt.ylabel('Potential Monthly Savings ($)')
            plt.title('Savings Potential by Optimization Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/savings_by_optimization_type.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Cost visualizations saved to: {output_dir}")


def main():
    """Example usage of cost analyzer"""
    
    # Initialize analyzer
    analyzer = CostAnalyzer()
    
    # Analyze costs for the last 30 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    logger.info("Starting cost analysis...")
    analysis = analyzer.analyze_costs(start_date, end_date)
    
    # Generate reports
    analyzer.generate_cost_report(analysis)
    analyzer.export_recommendations_csv(analysis['recommendations'])
    analyzer.visualize_costs(analysis)
    
    # Print summary
    print(f"\nCost Analysis Summary")
    print(f"=" * 50)
    print(f"Total Monthly Cost: ${analysis['total_cost']:,.2f}")
    print(f"Potential Savings: ${analysis['potential_savings']:,.2f}")
    print(f"Number of Recommendations: {len(analysis['recommendations'])}")
    print(f"\nTop 3 Recommendations:")
    
    for i, rec in enumerate(analysis['recommendations'][:3], 1):
        print(f"\n{i}. {rec.optimization_type.value.replace('_', ' ').title()}")
        print(f"   Resource: {rec.resource_id}")
        print(f"   Savings: ${rec.savings:,.2f}/month ({rec.savings_percentage:.1f}%)")
        print(f"   Risk: {rec.risk_level}, Effort: {rec.effort_level}")


if __name__ == "__main__":
    main()