# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "${var.project_name}-${var.environment}-db-subnet-group"
  }
}

# RDS Parameter Group
resource "aws_db_parameter_group" "main" {
  family = "postgres15"
  name   = "${var.project_name}-${var.environment}-db-params"

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-db-params"
  }
}

# RDS Instance for MLflow
resource "aws_db_instance" "mlflow" {
  identifier = "${var.project_name}-${var.environment}-mlflow-db"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.rds_instance_class

  db_name  = "mlflow"
  username = "mlflow_user"
  password = random_password.mlflow_db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  parameter_group_name   = aws_db_parameter_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  performance_insights_enabled = false
  monitoring_interval         = 60
  monitoring_role_arn         = aws_iam_role.rds_monitoring.arn

  tags = {
    Name    = "${var.project_name}-${var.environment}-mlflow-db"
    Purpose = "MLflow metadata storage"
  }
}

# RDS Instance for Airflow
resource "aws_db_instance" "airflow" {
  identifier = "${var.project_name}-${var.environment}-airflow-db"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.rds_instance_class

  db_name  = "airflow"
  username = "airflow_user"
  password = random_password.airflow_db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  parameter_group_name   = aws_db_parameter_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  performance_insights_enabled = false
  monitoring_interval         = 60
  monitoring_role_arn         = aws_iam_role.rds_monitoring.arn

  tags = {
    Name    = "${var.project_name}-${var.environment}-airflow-db"
    Purpose = "Airflow metadata storage"
  }
}

# Random passwords for RDS instances
resource "random_password" "mlflow_db_password" {
  length  = 16
  special = true
}

resource "random_password" "airflow_db_password" {
  length  = 16
  special = true
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.project_name}-${var.environment}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Store database credentials in AWS Secrets Manager
resource "aws_secretsmanager_secret" "mlflow_db" {
  name        = "${var.project_name}/${var.environment}/mlflow/database"
  description = "MLflow database credentials"

  tags = {
    Name = "${var.project_name}-${var.environment}-mlflow-db-secret"
  }
}

resource "aws_secretsmanager_secret_version" "mlflow_db" {
  secret_id = aws_secretsmanager_secret.mlflow_db.id
  secret_string = jsonencode({
    username = aws_db_instance.mlflow.username
    password = random_password.mlflow_db_password.result
    endpoint = aws_db_instance.mlflow.endpoint
    port     = aws_db_instance.mlflow.port
    dbname   = aws_db_instance.mlflow.db_name
  })
}

resource "aws_secretsmanager_secret" "airflow_db" {
  name        = "${var.project_name}/${var.environment}/airflow/database"
  description = "Airflow database credentials"

  tags = {
    Name = "${var.project_name}-${var.environment}-airflow-db-secret"
  }
}

resource "aws_secretsmanager_secret_version" "airflow_db" {
  secret_id = aws_secretsmanager_secret.airflow_db.id
  secret_string = jsonencode({
    username = aws_db_instance.airflow.username
    password = random_password.airflow_db_password.result
    endpoint = aws_db_instance.airflow.endpoint
    port     = aws_db_instance.airflow.port
    dbname   = aws_db_instance.airflow.db_name
  })
}