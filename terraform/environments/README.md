# Environment-Specific Configurations

This directory contains environment-specific Terraform variable files for different deployment environments.

## Available Environments

### Development (`dev.tfvars`)
- **Purpose**: Development and testing
- **CIDR**: 10.0.0.0/16
- **NAT Gateway**: Disabled (cost optimization)
- **RDS**: db.t3.micro
- **Features**: Minimal resource allocation for development work

### Staging (`staging.tfvars`)
- **Purpose**: Pre-production testing and validation
- **CIDR**: 10.1.0.0/16
- **NAT Gateway**: Enabled (production-like)
- **RDS**: db.t3.small
- **Features**: Production-like setup for thorough testing

### Production (`prod.tfvars`)
- **Purpose**: Live production workloads
- **CIDR**: 10.2.0.0/16
- **NAT Gateway**: Enabled (required)
- **RDS**: db.t3.medium
- **Features**: Production-grade instances and configurations

## Usage

Deploy to a specific environment using:

```bash
# Development
terraform plan -var-file="environments/dev.tfvars"
terraform apply -var-file="environments/dev.tfvars"

# Staging
terraform plan -var-file="environments/staging.tfvars"
terraform apply -var-file="environments/staging.tfvars"

# Production
terraform plan -var-file="environments/prod.tfvars"
terraform apply -var-file="environments/prod.tfvars"
```

## Environment Isolation

Each environment uses:
- **Separate CIDR blocks** to prevent IP conflicts
- **Environment-prefixed resource names** for clear identification
- **Environment-specific configurations** optimized for their use case
- **Separate state files** (when using remote state)

## Multi-Environment Deployment Strategy

1. **Workspace-based approach** (recommended):
   ```bash
   terraform workspace new dev
   terraform workspace select dev
   terraform apply -var-file="environments/dev.tfvars"
   
   terraform workspace new staging
   terraform workspace select staging
   terraform apply -var-file="environments/staging.tfvars"
   ```

2. **Directory-based approach**:
   ```bash
   # Copy terraform files to environment-specific directories
   cp -r . ../dev/
   cd ../dev/
   terraform apply -var-file="../terraform/environments/dev.tfvars"
   ```

## Cost Optimization by Environment

### Development
- NAT Gateway disabled (saves ~$45/month)
- Smallest RDS instances
- Minimal EKS node count

### Staging
- Production-like setup for accurate testing
- Medium-sized instances
- Full monitoring enabled

### Production
- High availability and redundancy
- Performance-optimized instances
- Enhanced monitoring and alerting
- Automated backups and disaster recovery

## Security Considerations

- Each environment has isolated networking (separate VPCs)
- Environment-specific IAM roles and policies
- Separate secrets management per environment
- Network access controls between environments