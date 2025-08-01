# Development Environment Configuration
aws_region = "us-west-2"
environment = "dev"

# Project Configuration
project_name = "mlops-drug-repurposing"

# Network Configuration - Using dev CIDR range
vpc_cidr = "10.0.0.0/16"
enable_nat_gateway = false  # Cost optimization for dev

# EKS Configuration - Smaller setup for development
eks_cluster_version = "1.27"

# RDS Configuration - Minimal size for development
rds_instance_class = "db.t3.micro"

# Tags for development resources
default_tags = {
  Environment = "dev"
  Project     = "MLOps Drug Repurposing"
  Owner       = "Development Team"
  CostCenter  = "R&D"
}