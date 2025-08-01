# Staging Environment Configuration
aws_region = "us-west-2"
environment = "staging"

# Project Configuration
project_name = "mlops-drug-repurposing"

# Network Configuration - Using staging CIDR range
vpc_cidr = "10.1.0.0/16"
enable_nat_gateway = true  # Enable for production-like testing

# EKS Configuration - Production-like setup
eks_cluster_version = "1.27"

# RDS Configuration - Medium size for staging
rds_instance_class = "db.t3.small"

# Tags for staging resources
default_tags = {
  Environment = "staging"
  Project     = "MLOps Drug Repurposing"
  Owner       = "QA Team"
  CostCenter  = "R&D"
}