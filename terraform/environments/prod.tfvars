# Production Environment Configuration
aws_region = "us-west-2"
environment = "prod"

# Project Configuration
project_name = "mlops-drug-repurposing"

# Network Configuration - Using production CIDR range
vpc_cidr = "10.2.0.0/16"
enable_nat_gateway = true  # Required for production

# EKS Configuration - Production setup
eks_cluster_version = "1.27"

# RDS Configuration - Production-grade instances
rds_instance_class = "db.t3.medium"

# Tags for production resources
default_tags = {
  Environment = "prod"
  Project     = "MLOps Drug Repurposing"
  Owner       = "Production Team"
  CostCenter  = "Operations"
  Backup      = "Required"
  Monitoring  = "Critical"
}