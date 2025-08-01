# MLOps Drug Repurposing - Terraform Infrastructure

This directory contains Terraform Infrastructure as Code (IaC) for the MLOps Drug Repurposing project on AWS.

## Architecture Overview

The infrastructure includes:
- **VPC** with public and private subnets across 2 AZs
- **EKS Cluster** for container orchestration
- **RDS PostgreSQL** instances for MLflow and Airflow metadata
- **S3 Buckets** for data, models, and artifacts storage
- **CloudWatch** monitoring and alerting
- **IAM Roles** and security groups for proper access control

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. kubectl installed (for EKS access)

## Quick Start

1. **Clone and navigate to terraform directory:**
   ```bash
   cd terraform
   ```

2. **Copy and customize variables:**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your preferred values
   ```

3. **Initialize Terraform:**
   ```bash
   terraform init
   ```

4. **Plan the deployment:**
   ```bash
   terraform plan
   ```

5. **Apply the infrastructure:**
   ```bash
   terraform apply
   ```

6. **Configure kubectl for EKS access:**
   ```bash
   aws eks update-kubeconfig --region us-west-2 --name mlops-drug-repurposing-cluster
   ```

## File Structure

```
terraform/
├── main.tf              # Provider configuration and data sources
├── variables.tf         # Input variables
├── vpc.tf              # VPC, subnets, routing
├── security.tf         # Security groups and IAM roles
├── s3.tf               # S3 buckets for data storage
├── eks.tf              # EKS cluster and node groups
├── rds.tf              # RDS instances and secrets
├── monitoring.tf       # CloudWatch monitoring setup
├── outputs.tf          # Output values
├── terraform.tfvars.example  # Example variables file
└── README.md           # This file
```

## Key Resources Created

### Networking
- VPC with 10.0.0.0/16 CIDR
- 2 public subnets (10.0.1.0/24, 10.0.2.0/24)
- 2 private subnets (10.0.10.0/24, 10.0.11.0/24)
- Internet Gateway and NAT Gateways

### Compute
- EKS cluster with managed node groups
- Auto-scaling configuration (1-10 nodes)
- t3.medium/large instance types

### Storage
- S3 buckets for data, models, and artifacts
- Versioning and encryption enabled
- Lifecycle policies for cost optimization

### Databases
- PostgreSQL RDS instances for MLflow and Airflow
- Encrypted storage and automated backups
- Credentials stored in AWS Secrets Manager

### Monitoring
- CloudWatch log groups and dashboard
- SNS topic for alerts
- Metric alarms for CPU, connections, and errors

## Outputs

After successful deployment, Terraform will output:
- VPC and subnet IDs
- EKS cluster endpoint and certificate
- S3 bucket names
- RDS endpoints (sensitive)
- Secret ARNs for database credentials
- CloudWatch dashboard URL

## Security Features

- All RDS instances use encryption at rest
- S3 buckets have public access blocked
- IAM roles follow least-privilege principle
- Security groups restrict access to necessary ports
- Database credentials stored in Secrets Manager

## Cost Optimization

- NAT Gateways can be disabled for development (set `enable_nat_gateway = false`)
- RDS instances use `db.t3.micro` by default
- S3 lifecycle policies automatically archive old data
- CloudWatch log retention periods configured

## Customization

Edit `terraform.tfvars` to customize:
- AWS region and environment name
- Instance sizes and cluster configuration
- Network CIDR blocks
- Enable/disable NAT Gateways for cost savings

## Cleanup

To destroy all resources:
```bash
terraform destroy
```

**Warning:** This will permanently delete all data in S3 buckets and RDS instances.

## Next Steps

After infrastructure deployment:
1. Configure kubectl access to EKS cluster
2. Deploy MLflow and Airflow using Helm charts
3. Set up CI/CD pipelines
4. Configure monitoring dashboards
5. Implement data pipelines

## Troubleshooting

Common issues:
- **EKS access denied**: Ensure AWS CLI is configured with correct permissions
- **RDS connection timeout**: Check security groups and subnet routing
- **S3 bucket name conflicts**: Bucket names must be globally unique
- **Region-specific resources**: Ensure all resources are in the same region

For more help, check the Terraform plan output and AWS CloudFormation events.