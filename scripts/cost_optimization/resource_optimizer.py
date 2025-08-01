#!/usr/bin/env python3
"""
Resource Optimizer

Implements cost optimization recommendations for cloud resources.
"""

import json
import logging
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import boto3
from botocore.exceptions import ClientError
import yaml
import schedule
import time as time_module
from dataclasses import dataclass
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of optimization actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationAction:
    """Represents an optimization action"""
    action_id: str
    resource_id: str
    resource_type: str
    action_type: str
    description: str
    estimated_savings: float
    status: ActionStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None


class ResourceOptimizer:
    """Implements resource optimization actions"""
    
    def __init__(self, 
                 aws_profile: Optional[str] = None,
                 region: str = "us-east-1",
                 config_path: str = "configs/cost_optimization_config.yaml",
                 dry_run: bool = False):
        """
        Initialize resource optimizer
        
        Args:
            aws_profile: AWS profile to use
            region: AWS region
            config_path: Path to configuration file
            dry_run: If True, only simulate actions
        """
        self.region = region
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.ec2_client = session.client('ec2', region_name=region)
        self.rds_client = session.client('rds', region_name=region)
        self.s3_client = session.client('s3', region_name=region)
        self.autoscaling_client = session.client('autoscaling', region_name=region)
        self.cloudwatch_client = session.client('cloudwatch', region_name=region)
        self.lambda_client = session.client('lambda', region_name=region)
        
        # Action history
        self.action_history: List[OptimizationAction] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def optimize_ec2_instance(self, 
                            instance_id: str,
                            target_instance_type: str,
                            schedule_stop: bool = False) -> OptimizationAction:
        """
        Optimize an EC2 instance
        
        Args:
            instance_id: EC2 instance ID
            target_instance_type: Target instance type for rightsizing
            schedule_stop: Whether to schedule stop/start
            
        Returns:
            Optimization action result
        """
        action = OptimizationAction(
            action_id=f"ec2_opt_{instance_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            resource_id=instance_id,
            resource_type="EC2",
            action_type="rightsizing",
            description=f"Resize instance to {target_instance_type}",
            estimated_savings=0,  # Will be calculated
            status=ActionStatus.IN_PROGRESS,
            created_at=datetime.utcnow()
        )
        
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would resize {instance_id} to {target_instance_type}")
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
            else:
                # Get current instance details
                response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
                instance = response['Reservations'][0]['Instances'][0]
                current_type = instance['InstanceType']
                
                # Store rollback info
                action.rollback_info = {'original_instance_type': current_type}
                
                # Stop instance if running
                if instance['State']['Name'] == 'running':
                    logger.info(f"Stopping instance {instance_id}")
                    self.ec2_client.stop_instances(InstanceIds=[instance_id])
                    
                    # Wait for instance to stop
                    waiter = self.ec2_client.get_waiter('instance_stopped')
                    waiter.wait(InstanceIds=[instance_id])
                
                # Modify instance type
                logger.info(f"Modifying instance type from {current_type} to {target_instance_type}")
                self.ec2_client.modify_instance_attribute(
                    InstanceId=instance_id,
                    InstanceType={'Value': target_instance_type}
                )
                
                # Start instance
                logger.info(f"Starting instance {instance_id}")
                self.ec2_client.start_instances(InstanceIds=[instance_id])
                
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
                
                # Set up scheduling if requested
                if schedule_stop:
                    self._setup_instance_scheduling(instance_id)
                    
        except ClientError as e:
            logger.error(f"Error optimizing EC2 instance {instance_id}: {str(e)}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
        
        self.action_history.append(action)
        return action
    
    def optimize_rds_instance(self,
                            db_instance_id: str,
                            target_instance_class: str,
                            enable_auto_scaling: bool = True) -> OptimizationAction:
        """
        Optimize an RDS instance
        
        Args:
            db_instance_id: RDS instance identifier
            target_instance_class: Target instance class
            enable_auto_scaling: Enable storage auto-scaling
            
        Returns:
            Optimization action result
        """
        action = OptimizationAction(
            action_id=f"rds_opt_{db_instance_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            resource_id=db_instance_id,
            resource_type="RDS",
            action_type="rightsizing",
            description=f"Resize RDS instance to {target_instance_class}",
            estimated_savings=0,
            status=ActionStatus.IN_PROGRESS,
            created_at=datetime.utcnow()
        )
        
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would resize RDS {db_instance_id} to {target_instance_class}")
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
            else:
                # Get current instance details
                response = self.rds_client.describe_db_instances(
                    DBInstanceIdentifier=db_instance_id
                )
                db_instance = response['DBInstances'][0]
                current_class = db_instance['DBInstanceClass']
                
                # Store rollback info
                action.rollback_info = {
                    'original_instance_class': current_class,
                    'original_max_allocated_storage': db_instance.get('MaxAllocatedStorage')
                }
                
                # Modify instance
                logger.info(f"Modifying RDS instance class from {current_class} to {target_instance_class}")
                
                modify_params = {
                    'DBInstanceIdentifier': db_instance_id,
                    'DBInstanceClass': target_instance_class,
                    'ApplyImmediately': False  # Apply during maintenance window
                }
                
                # Enable storage auto-scaling if requested
                if enable_auto_scaling and not db_instance.get('MaxAllocatedStorage'):
                    current_storage = db_instance['AllocatedStorage']
                    modify_params['MaxAllocatedStorage'] = min(current_storage * 3, 1000)
                    logger.info(f"Enabling storage auto-scaling up to {modify_params['MaxAllocatedStorage']} GB")
                
                self.rds_client.modify_db_instance(**modify_params)
                
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
                
        except ClientError as e:
            logger.error(f"Error optimizing RDS instance {db_instance_id}: {str(e)}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
        
        self.action_history.append(action)
        return action
    
    def optimize_s3_bucket(self, 
                          bucket_name: str,
                          enable_lifecycle: bool = True,
                          enable_intelligent_tiering: bool = True) -> OptimizationAction:
        """
        Optimize an S3 bucket
        
        Args:
            bucket_name: S3 bucket name
            enable_lifecycle: Enable lifecycle policies
            enable_intelligent_tiering: Enable intelligent tiering
            
        Returns:
            Optimization action result
        """
        action = OptimizationAction(
            action_id=f"s3_opt_{bucket_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            resource_id=bucket_name,
            resource_type="S3",
            action_type="storage_optimization",
            description=f"Optimize S3 bucket {bucket_name}",
            estimated_savings=0,
            status=ActionStatus.IN_PROGRESS,
            created_at=datetime.utcnow()
        )
        
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would optimize S3 bucket {bucket_name}")
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
            else:
                # Enable lifecycle policies
                if enable_lifecycle:
                    lifecycle_config = self._create_s3_lifecycle_config()
                    logger.info(f"Applying lifecycle policy to {bucket_name}")
                    
                    self.s3_client.put_bucket_lifecycle_configuration(
                        Bucket=bucket_name,
                        LifecycleConfiguration=lifecycle_config
                    )
                
                # Enable intelligent tiering
                if enable_intelligent_tiering:
                    logger.info(f"Enabling intelligent tiering for {bucket_name}")
                    
                    tiering_config = {
                        'Id': 'optimize-all-objects',
                        'Status': 'Enabled',
                        'Tierings': [
                            {
                                'Days': 90,
                                'AccessTier': 'ARCHIVE_ACCESS'
                            },
                            {
                                'Days': 180,
                                'AccessTier': 'DEEP_ARCHIVE_ACCESS'
                            }
                        ]
                    }
                    
                    self.s3_client.put_bucket_intelligent_tiering_configuration(
                        Bucket=bucket_name,
                        Id='optimize-all-objects',
                        IntelligentTieringConfiguration=tiering_config
                    )
                
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.utcnow()
                
        except ClientError as e:
            logger.error(f"Error optimizing S3 bucket {bucket_name}: {str(e)}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
        
        self.action_history.append(action)
        return action
    
    def _create_s3_lifecycle_config(self) -> Dict[str, Any]:
        """Create S3 lifecycle configuration based on config"""
        config = self.config['cost_optimization']['services']['s3']
        lifecycle_days = self.config['cost_optimization']['optimization']['storage']['s3_lifecycle_days']
        
        return {
            'Rules': [
                {
                    'ID': 'transition-to-ia',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': lifecycle_days['standard_to_ia'],
                            'StorageClass': 'STANDARD_IA'
                        }
                    ]
                },
                {
                    'ID': 'transition-to-glacier',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': lifecycle_days['ia_to_glacier'],
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': 'transition-to-deep-archive',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': lifecycle_days['glacier_to_deep'],
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ]
                },
                {
                    'ID': 'delete-incomplete-multipart-uploads',
                    'Status': 'Enabled',
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': 7
                    }
                }
            ]
        }
    
    def _setup_instance_scheduling(self, instance_id: str) -> None:
        """Set up instance scheduling using tags and Lambda"""
        scheduling_config = self.config['cost_optimization']['optimization']['scheduling']
        
        # Tag instance with scheduling information
        tags = [
            {
                'Key': 'ScheduleStart',
                'Value': f"{scheduling_config['business_hours_start']}:00"
            },
            {
                'Key': 'ScheduleStop',
                'Value': f"{scheduling_config['business_hours_end']}:00"
            },
            {
                'Key': 'ScheduleTimezone',
                'Value': scheduling_config['timezone']
            },
            {
                'Key': 'ScheduleDays',
                'Value': ','.join(map(str, scheduling_config['business_days']))
            }
        ]
        
        try:
            self.ec2_client.create_tags(
                Resources=[instance_id],
                Tags=tags
            )
            logger.info(f"Applied scheduling tags to instance {instance_id}")
        except ClientError as e:
            logger.error(f"Error applying scheduling tags: {str(e)}")
    
    def setup_auto_scaling(self,
                          resource_id: str,
                          resource_type: str,
                          min_capacity: int,
                          max_capacity: int,
                          target_utilization: int = 70) -> OptimizationAction:
        """
        Set up auto-scaling for a resource
        
        Args:
            resource_id: Resource identifier
            resource_type: Type of resource (EC2, RDS, etc.)
            min_capacity: Minimum capacity
            max_capacity: Maximum capacity
            target_utilization: Target CPU utilization percentage
            
        Returns:
            Optimization action result
        """
        action = OptimizationAction(
            action_id=f"autoscale_{resource_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            resource_id=resource_id,
            resource_type=resource_type,
            action_type="auto_scaling",
            description=f"Set up auto-scaling for {resource_id}",
            estimated_savings=0,
            status=ActionStatus.IN_PROGRESS,
            created_at=datetime.utcnow()
        )
        
        try:
            if resource_type == "EC2" and not self.dry_run:
                # Create or update auto-scaling group
                self._setup_ec2_auto_scaling(
                    resource_id, min_capacity, max_capacity, target_utilization
                )
            elif resource_type == "RDS" and not self.dry_run:
                # RDS already has storage auto-scaling, this would be for read replicas
                logger.info(f"RDS auto-scaling setup for {resource_id}")
                # Implementation depends on specific requirements
            
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error setting up auto-scaling: {str(e)}")
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
        
        self.action_history.append(action)
        return action
    
    def _setup_ec2_auto_scaling(self,
                               instance_id: str,
                               min_capacity: int,
                               max_capacity: int,
                               target_utilization: int) -> None:
        """Set up EC2 auto-scaling"""
        # This is a simplified version - actual implementation would be more complex
        
        # Create launch template from instance
        response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        launch_template_name = f"lt-{instance_id}"
        
        # Create launch template
        self.ec2_client.create_launch_template(
            LaunchTemplateName=launch_template_name,
            LaunchTemplateData={
                'InstanceType': instance['InstanceType'],
                'ImageId': instance['ImageId'],
                'SecurityGroupIds': [sg['GroupId'] for sg in instance['SecurityGroups']],
                'KeyName': instance.get('KeyName', ''),
                'IamInstanceProfile': instance.get('IamInstanceProfile', {})
            }
        )
        
        # Create auto-scaling group
        asg_name = f"asg-{instance_id}"
        
        self.autoscaling_client.create_auto_scaling_group(
            AutoScalingGroupName=asg_name,
            LaunchTemplate={
                'LaunchTemplateName': launch_template_name,
                'Version': '$Latest'
            },
            MinSize=min_capacity,
            MaxSize=max_capacity,
            DesiredCapacity=min_capacity,
            DefaultCooldown=300,
            AvailabilityZones=[instance['Placement']['AvailabilityZone']],
            HealthCheckType='EC2',
            HealthCheckGracePeriod=300
        )
        
        # Create scaling policy
        self.autoscaling_client.put_scaling_policy(
            AutoScalingGroupName=asg_name,
            PolicyName=f"{asg_name}-target-tracking",
            PolicyType='TargetTrackingScaling',
            TargetTrackingConfiguration={
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'ASGAverageCPUUtilization'
                },
                'TargetValue': float(target_utilization)
            }
        )
        
        logger.info(f"Created auto-scaling group {asg_name} for instance {instance_id}")
    
    def clean_up_unused_resources(self) -> List[OptimizationAction]:
        """Clean up unused resources based on configuration"""
        actions = []
        automation_config = self.config['cost_optimization']['automation']
        
        if automation_config['actions']['stop_idle_instances']['enabled']:
            actions.extend(self._stop_idle_instances())
        
        if automation_config['actions']['delete_unused_volumes']['enabled']:
            actions.extend(self._delete_unused_volumes())
        
        if automation_config['actions']['delete_old_snapshots']['enabled']:
            actions.extend(self._delete_old_snapshots())
        
        if automation_config['actions']['remove_unused_elastic_ips']['enabled']:
            actions.extend(self._remove_unused_elastic_ips())
        
        return actions
    
    def _stop_idle_instances(self) -> List[OptimizationAction]:
        """Stop idle EC2 instances"""
        actions = []
        config = self.config['cost_optimization']['automation']['actions']['stop_idle_instances']
        idle_days = config['idle_days']
        excluded_tags = config.get('excluded_tags', {})
        
        try:
            # Get all running instances
            response = self.ec2_client.describe_instances(
                Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    # Check excluded tags
                    instance_tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    
                    excluded = False
                    for tag_key, tag_value in excluded_tags.items():
                        if instance_tags.get(tag_key) == tag_value:
                            excluded = True
                            break
                    
                    if excluded:
                        continue
                    
                    # Check if instance is idle (simplified check)
                    # In real implementation, would check CloudWatch metrics
                    action = OptimizationAction(
                        action_id=f"stop_idle_{instance['InstanceId']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_id=instance['InstanceId'],
                        resource_type="EC2",
                        action_type="stop_idle",
                        description=f"Stop idle instance {instance['InstanceId']}",
                        estimated_savings=0,  # Calculate based on instance type
                        status=ActionStatus.PENDING,
                        created_at=datetime.utcnow()
                    )
                    
                    if not self.dry_run:
                        self.ec2_client.stop_instances(InstanceIds=[instance['InstanceId']])
                        action.status = ActionStatus.COMPLETED
                        action.completed_at = datetime.utcnow()
                    else:
                        logger.info(f"[DRY RUN] Would stop idle instance {instance['InstanceId']}")
                        action.status = ActionStatus.COMPLETED
                    
                    actions.append(action)
                    self.action_history.append(action)
                    
        except ClientError as e:
            logger.error(f"Error stopping idle instances: {str(e)}")
        
        return actions
    
    def _delete_unused_volumes(self) -> List[OptimizationAction]:
        """Delete unused EBS volumes"""
        actions = []
        config = self.config['cost_optimization']['automation']['actions']['delete_unused_volumes']
        unused_days = config['unused_days']
        
        try:
            # Get all available (unattached) volumes
            response = self.ec2_client.describe_volumes(
                Filters=[{'Name': 'status', 'Values': ['available']}]
            )
            
            for volume in response['Volumes']:
                # Check how long volume has been unattached
                # In real implementation, would track this over time
                
                action = OptimizationAction(
                    action_id=f"delete_vol_{volume['VolumeId']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    resource_id=volume['VolumeId'],
                    resource_type="EBS",
                    action_type="delete_unused",
                    description=f"Delete unused volume {volume['VolumeId']}",
                    estimated_savings=volume['Size'] * 0.10,  # Estimate $0.10/GB/month
                    status=ActionStatus.PENDING,
                    created_at=datetime.utcnow()
                )
                
                if not self.dry_run:
                    # Create snapshot before deletion
                    snapshot_response = self.ec2_client.create_snapshot(
                        VolumeId=volume['VolumeId'],
                        Description=f"Backup before deletion - {datetime.utcnow().isoformat()}"
                    )
                    
                    action.rollback_info = {'snapshot_id': snapshot_response['SnapshotId']}
                    
                    # Delete volume
                    self.ec2_client.delete_volume(VolumeId=volume['VolumeId'])
                    action.status = ActionStatus.COMPLETED
                    action.completed_at = datetime.utcnow()
                else:
                    logger.info(f"[DRY RUN] Would delete unused volume {volume['VolumeId']}")
                    action.status = ActionStatus.COMPLETED
                
                actions.append(action)
                self.action_history.append(action)
                
        except ClientError as e:
            logger.error(f"Error deleting unused volumes: {str(e)}")
        
        return actions
    
    def _delete_old_snapshots(self) -> List[OptimizationAction]:
        """Delete old EBS snapshots"""
        actions = []
        config = self.config['cost_optimization']['automation']['actions']['delete_old_snapshots']
        retention_days = config['retention_days']
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        try:
            # Get all snapshots owned by the account
            response = self.ec2_client.describe_snapshots(OwnerIds=['self'])
            
            for snapshot in response['Snapshots']:
                start_time = snapshot['StartTime'].replace(tzinfo=None)
                
                if start_time < cutoff_date:
                    action = OptimizationAction(
                        action_id=f"delete_snap_{snapshot['SnapshotId']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_id=snapshot['SnapshotId'],
                        resource_type="EBS_SNAPSHOT",
                        action_type="delete_old",
                        description=f"Delete old snapshot {snapshot['SnapshotId']}",
                        estimated_savings=snapshot['VolumeSize'] * 0.05,  # $0.05/GB/month
                        status=ActionStatus.PENDING,
                        created_at=datetime.utcnow()
                    )
                    
                    if not self.dry_run:
                        self.ec2_client.delete_snapshot(SnapshotId=snapshot['SnapshotId'])
                        action.status = ActionStatus.COMPLETED
                        action.completed_at = datetime.utcnow()
                    else:
                        logger.info(f"[DRY RUN] Would delete old snapshot {snapshot['SnapshotId']}")
                        action.status = ActionStatus.COMPLETED
                    
                    actions.append(action)
                    self.action_history.append(action)
                    
        except ClientError as e:
            logger.error(f"Error deleting old snapshots: {str(e)}")
        
        return actions
    
    def _remove_unused_elastic_ips(self) -> List[OptimizationAction]:
        """Remove unused Elastic IPs"""
        actions = []
        
        try:
            # Get all Elastic IPs
            response = self.ec2_client.describe_addresses()
            
            for address in response['Addresses']:
                if 'InstanceId' not in address and 'NetworkInterfaceId' not in address:
                    # Elastic IP is not associated with any resource
                    action = OptimizationAction(
                        action_id=f"release_eip_{address['AllocationId']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        resource_id=address['AllocationId'],
                        resource_type="ELASTIC_IP",
                        action_type="release_unused",
                        description=f"Release unused Elastic IP {address.get('PublicIp', 'Unknown')}",
                        estimated_savings=3.60,  # $0.005/hour * 24 * 30
                        status=ActionStatus.PENDING,
                        created_at=datetime.utcnow()
                    )
                    
                    if not self.dry_run:
                        self.ec2_client.release_address(AllocationId=address['AllocationId'])
                        action.status = ActionStatus.COMPLETED
                        action.completed_at = datetime.utcnow()
                    else:
                        logger.info(f"[DRY RUN] Would release Elastic IP {address.get('PublicIp', 'Unknown')}")
                        action.status = ActionStatus.COMPLETED
                    
                    actions.append(action)
                    self.action_history.append(action)
                    
        except ClientError as e:
            logger.error(f"Error removing unused Elastic IPs: {str(e)}")
        
        return actions
    
    def rollback_action(self, action_id: str) -> bool:
        """
        Rollback a completed optimization action
        
        Args:
            action_id: Action ID to rollback
            
        Returns:
            Success status
        """
        # Find the action
        action = None
        for a in self.action_history:
            if a.action_id == action_id:
                action = a
                break
        
        if not action:
            logger.error(f"Action {action_id} not found")
            return False
        
        if action.status != ActionStatus.COMPLETED:
            logger.error(f"Action {action_id} is not in completed state")
            return False
        
        if not action.rollback_info:
            logger.error(f"No rollback information for action {action_id}")
            return False
        
        try:
            if action.resource_type == "EC2" and action.action_type == "rightsizing":
                # Rollback EC2 instance type change
                original_type = action.rollback_info['original_instance_type']
                self.optimize_ec2_instance(action.resource_id, original_type)
                
            elif action.resource_type == "RDS" and action.action_type == "rightsizing":
                # Rollback RDS instance class change
                original_class = action.rollback_info['original_instance_class']
                self.optimize_rds_instance(action.resource_id, original_class, enable_auto_scaling=False)
                
            elif action.resource_type == "EBS" and action.action_type == "delete_unused":
                # Restore volume from snapshot
                snapshot_id = action.rollback_info['snapshot_id']
                self.ec2_client.create_volume(
                    SnapshotId=snapshot_id,
                    AvailabilityZone=self.region + 'a'  # Simplified - should get proper AZ
                )
            
            action.status = ActionStatus.ROLLED_BACK
            logger.info(f"Successfully rolled back action {action_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back action {action_id}: {str(e)}")
            return False
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate report of optimization actions"""
        completed_actions = [a for a in self.action_history if a.status == ActionStatus.COMPLETED]
        failed_actions = [a for a in self.action_history if a.status == ActionStatus.FAILED]
        
        total_savings = sum(a.estimated_savings for a in completed_actions)
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_actions': len(self.action_history),
                'completed_actions': len(completed_actions),
                'failed_actions': len(failed_actions),
                'total_monthly_savings': total_savings,
                'total_yearly_savings': total_savings * 12
            },
            'actions_by_type': {},
            'actions_by_resource': {},
            'failed_actions': [
                {
                    'action_id': a.action_id,
                    'resource_id': a.resource_id,
                    'error': a.error_message
                }
                for a in failed_actions
            ]
        }
        
        # Group by action type
        for action in completed_actions:
            if action.action_type not in report['actions_by_type']:
                report['actions_by_type'][action.action_type] = {
                    'count': 0,
                    'savings': 0
                }
            report['actions_by_type'][action.action_type]['count'] += 1
            report['actions_by_type'][action.action_type]['savings'] += action.estimated_savings
        
        # Group by resource type
        for action in completed_actions:
            if action.resource_type not in report['actions_by_resource']:
                report['actions_by_resource'][action.resource_type] = {
                    'count': 0,
                    'savings': 0
                }
            report['actions_by_resource'][action.resource_type]['count'] += 1
            report['actions_by_resource'][action.resource_type]['savings'] += action.estimated_savings
        
        return report


def main():
    """Example usage of resource optimizer"""
    
    # Initialize optimizer
    optimizer = ResourceOptimizer(dry_run=True)  # Use dry_run for safety
    
    # Example: Optimize an EC2 instance
    print("Optimizing EC2 instance...")
    action1 = optimizer.optimize_ec2_instance(
        instance_id="i-1234567890abcdef0",
        target_instance_type="t3.small",
        schedule_stop=True
    )
    print(f"Action: {action1.description}, Status: {action1.status.value}")
    
    # Example: Clean up unused resources
    print("\nCleaning up unused resources...")
    cleanup_actions = optimizer.clean_up_unused_resources()
    print(f"Performed {len(cleanup_actions)} cleanup actions")
    
    # Generate report
    print("\nGenerating optimization report...")
    report = optimizer.generate_optimization_report()
    
    print(f"\nOptimization Summary:")
    print(f"Total actions: {report['summary']['total_actions']}")
    print(f"Completed: {report['summary']['completed_actions']}")
    print(f"Failed: {report['summary']['failed_actions']}")
    print(f"Estimated monthly savings: ${report['summary']['total_monthly_savings']:,.2f}")
    print(f"Estimated yearly savings: ${report['summary']['total_yearly_savings']:,.2f}")


if __name__ == "__main__":
    main()