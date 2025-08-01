#!/usr/bin/env python3
"""
Model Approval Workflows
Comprehensive approval workflow system for model lifecycle management with multi-tier approval,
emergency procedures, and automated governance.
"""

import argparse
import json
import sys
import time
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import yaml
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import uuid
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status types."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    AUTO_APPROVED = "AUTO_APPROVED"
    EMERGENCY_APPROVED = "EMERGENCY_APPROVED"


class ApprovalType(Enum):
    """Types of approval requests."""
    STAGE_TRANSITION = "STAGE_TRANSITION"
    MODEL_DEPLOYMENT = "MODEL_DEPLOYMENT"
    PRODUCTION_PROMOTION = "PRODUCTION_PROMOTION"
    EMERGENCY_DEPLOYMENT = "EMERGENCY_DEPLOYMENT"
    MODEL_ARCHIVAL = "MODEL_ARCHIVAL"
    ROLLBACK = "ROLLBACK"


class ApprovalUrgency(Enum):
    """Approval urgency levels."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class ApprovalRequest:
    """Approval request structure."""
    request_id: str
    approval_type: str
    model_name: str
    model_version: str
    from_stage: Optional[str]
    to_stage: str
    requested_by: str
    request_timestamp: str
    expiry_timestamp: str
    urgency: str
    reason: str
    description: str
    
    # Approval details
    required_approvers: List[str]
    min_approvals: int
    approvals_received: List[Dict[str, Any]]
    
    # Status and results
    status: str
    final_decision: Optional[str]
    decision_timestamp: Optional[str]
    decision_reason: Optional[str]
    
    # Supporting data
    validation_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    # Compliance and audit
    compliance_checks: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    
    # Emergency provisions
    is_emergency: bool
    emergency_justification: Optional[str]
    emergency_approver: Optional[str]


@dataclass
class ApprovalResponse:
    """Individual approval response."""
    response_id: str
    request_id: str
    approver: str
    decision: str  # APPROVE, REJECT, ABSTAIN
    timestamp: str
    comments: str
    conditions: List[str]
    digital_signature: Optional[str]


class ApprovalWorkflowManager:
    """Manages model approval workflows and governance."""
    
    def __init__(self, config_path: str = "configs/approval_workflows_config.yaml"):
        """Initialize approval workflow manager.
        
        Args:
            config_path: Path to approval workflow configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.mlflow_client = MlflowClient()
        
        # Request tracking
        self.pending_requests = {}
        self.completed_requests = {}
        self.approval_history = []
        
        # Notification system
        self.notification_manager = NotificationManager(self.config)
        
        # Digital signature system
        self.signature_manager = DigitalSignatureManager(self.config)
        
        # Compliance validator
        self.compliance_validator = ComplianceValidator(self.config)
        
        # Emergency procedures
        self.emergency_manager = EmergencyApprovalManager(self.config)
        
        # Setup approval tracking experiment
        self._setup_mlflow_tracking()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load approval workflow configuration."""
        default_config = {
            "approval_policies": {
                "stage_transition": {
                    "staging": {
                        "required": False,
                        "approvers": [],
                        "min_approvals": 0,
                        "timeout_hours": 24
                    },
                    "production": {
                        "required": True,
                        "approvers": ["model-admin", "ml-ops-lead", "data-science-lead"],
                        "min_approvals": 2,
                        "timeout_hours": 72
                    },
                    "archived": {
                        "required": False,
                        "approvers": [],
                        "min_approvals": 0,
                        "timeout_hours": 24
                    }
                },
                "emergency_deployment": {
                    "required": True,
                    "approvers": ["cto", "head-of-engineering", "security-lead"],
                    "min_approvals": 1,
                    "timeout_hours": 4,
                    "valid_reasons": [
                        "security_vulnerability",
                        "critical_bug_fix",
                        "regulatory_compliance",
                        "data_breach_response",
                        "system_outage"
                    ]
                }
            },
            "notification_config": {
                "channels": ["slack", "email"],
                "templates": {
                    "approval_request": {
                        "subject": "Model Approval Required: {model_name} v{version}",
                        "urgency_escalation": True
                    }
                }
            },
            "compliance_config": {
                "electronic_signatures": True,
                "audit_trail_required": True,
                "validation_documentation": True,
                "change_control": True,
                "fda_21_cfr_part_11": True
            },
            "auto_approval_rules": {
                "enabled": True,
                "conditions": {
                    "trusted_users": ["ci-bot", "automated-system"],
                    "non_production_stages": ["staging", "development"],
                    "performance_improvement_threshold": 0.05,
                    "security_scan_passed": True
                }
            }
        }
        
        try:
            import yaml
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return {**default_config, **config}
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return default_config
        except ImportError:
            logger.warning("PyYAML not installed, using default configuration")
            return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return default_config
    
    def _setup_mlflow_tracking(self):
        """Setup MLflow tracking for approval workflows."""
        try:
            experiment_name = "model_approval_workflows"
            try:
                mlflow.get_experiment_by_name(experiment_name)
            except:
                mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    def create_approval_request(self, model_name: str, model_version: str,
                              to_stage: str, from_stage: Optional[str] = None,
                              approval_type: ApprovalType = ApprovalType.STAGE_TRANSITION,
                              reason: str = "", description: str = "",
                              urgency: ApprovalUrgency = ApprovalUrgency.NORMAL,
                              is_emergency: bool = False,
                              emergency_justification: str = "") -> str:
        """Create a new approval request.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            to_stage: Target stage
            from_stage: Source stage (if applicable)
            approval_type: Type of approval request
            reason: Reason for the request
            description: Detailed description
            urgency: Urgency level
            is_emergency: Whether this is an emergency request
            emergency_justification: Justification for emergency
            
        Returns:
            Request ID
        """
        logger.info(f"Creating approval request for {model_name} v{model_version} -> {to_stage}")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get approval policy for the target stage
        policy = self._get_approval_policy(approval_type, to_stage, is_emergency)
        
        if not policy["required"]:
            logger.info("Approval not required, auto-approving")
            return self._auto_approve_request(model_name, model_version, to_stage, reason)
        
        # Calculate expiry time
        timeout_hours = policy["timeout_hours"]
        expiry_time = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)
        
        # Get model performance and validation data
        validation_results = self._get_validation_results(model_name, model_version)
        performance_metrics = self._get_performance_metrics(model_name, model_version)
        
        # Perform impact and risk assessment
        impact_assessment = self._assess_impact(model_name, model_version, to_stage)
        risk_assessment = self._assess_risk(model_name, model_version, to_stage)
        
        # Compliance checks
        compliance_checks = self.compliance_validator.validate_compliance(
            model_name, model_version, to_stage
        )
        
        # Create approval request
        approval_request = ApprovalRequest(
            request_id=request_id,
            approval_type=approval_type.value,
            model_name=model_name,
            model_version=model_version,
            from_stage=from_stage,
            to_stage=to_stage,
            requested_by=os.getenv("USER", "system"),
            request_timestamp=datetime.now(timezone.utc).isoformat(),
            expiry_timestamp=expiry_time.isoformat(),
            urgency=urgency.value,
            reason=reason,
            description=description,
            
            required_approvers=policy["approvers"],
            min_approvals=policy["min_approvals"],
            approvals_received=[],
            
            status=ApprovalStatus.PENDING.value,
            final_decision=None,
            decision_timestamp=None,
            decision_reason=None,
            
            validation_results=validation_results,
            performance_metrics=performance_metrics,
            impact_assessment=impact_assessment,
            risk_assessment=risk_assessment,
            
            compliance_checks=compliance_checks,
            audit_trail=[{
                "action": "request_created",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": os.getenv("USER", "system"),
                "details": {"reason": reason, "urgency": urgency.value}
            }],
            
            is_emergency=is_emergency,
            emergency_justification=emergency_justification if is_emergency else None,
            emergency_approver=None
        )
        
        # Check for auto-approval conditions
        if self._check_auto_approval_conditions(approval_request):
            return self._auto_approve_request(model_name, model_version, to_stage, reason, request_id)
        
        # Store the request
        self.pending_requests[request_id] = approval_request
        
        # Send notifications
        self.notification_manager.send_approval_request_notification(approval_request)
        
        # Log to MLflow
        self._log_approval_request_to_mlflow(approval_request)
        
        logger.info(f"Approval request created: {request_id}")
        return request_id
    
    def submit_approval_response(self, request_id: str, approver: str,
                               decision: str, comments: str = "",
                               conditions: List[str] = None) -> bool:
        """Submit an approval response.
        
        Args:
            request_id: Request ID
            approver: Name of the approver
            decision: APPROVE, REJECT, or ABSTAIN
            comments: Approval comments
            conditions: Any conditions for approval
            
        Returns:
            Success status
        """
        logger.info(f"Processing approval response from {approver} for {request_id}")
        
        if request_id not in self.pending_requests:
            if request_id in self.completed_requests:
                logger.warning(f"Request {request_id} already completed")
                return False
            else:
                logger.error(f"Request {request_id} not found")
                return False
        
        approval_request = self.pending_requests[request_id]
        
        # Validate approver
        if approver not in approval_request.required_approvers:
            logger.error(f"User {approver} not authorized to approve this request")
            return False
        
        # Check if already responded
        for existing_response in approval_request.approvals_received:
            if existing_response["approver"] == approver:
                logger.warning(f"Approver {approver} already responded to this request")
                return False
        
        # Check if request expired
        if datetime.now(timezone.utc) > datetime.fromisoformat(approval_request.expiry_timestamp):
            logger.warning(f"Request {request_id} has expired")
            approval_request.status = ApprovalStatus.EXPIRED.value
            self._finalize_request(request_id, "EXPIRED", "Request expired without sufficient approvals")
            return False
        
        # Create approval response
        response_id = str(uuid.uuid4())
        digital_signature = self.signature_manager.create_signature(
            approver, decision, comments, approval_request
        )
        
        approval_response = ApprovalResponse(
            response_id=response_id,
            request_id=request_id,
            approver=approver,
            decision=decision,
            timestamp=datetime.now(timezone.utc).isoformat(),
            comments=comments,
            conditions=conditions or [],
            digital_signature=digital_signature
        )
        
        # Add response to request
        approval_request.approvals_received.append(asdict(approval_response))
        
        # Update audit trail
        approval_request.audit_trail.append({
            "action": "approval_response_received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": approver,
            "details": {
                "decision": decision,
                "comments": comments,
                "conditions": conditions or []
            }
        })
        
        # Check if decision can be made
        decision_result = self._evaluate_approval_decision(approval_request)
        
        if decision_result["final_decision"]:
            self._finalize_request(
                request_id, 
                decision_result["final_decision"],
                decision_result["reason"]
            )
        
        # Send notification
        self.notification_manager.send_approval_response_notification(approval_request, approval_response)
        
        # Log to MLflow
        self._log_approval_response_to_mlflow(approval_request, approval_response)
        
        logger.info(f"Approval response processed successfully")
        return True
    
    def get_approval_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get approval request status.
        
        Args:
            request_id: Request ID
            
        Returns:
            Approval status information
        """
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
        elif request_id in self.completed_requests:
            request = self.completed_requests[request_id]
        else:
            return None
        
        # Calculate approval progress
        approvals_count = len([r for r in request.approvals_received if r["decision"] == "APPROVE"])
        rejections_count = len([r for r in request.approvals_received if r["decision"] == "REJECT"])
        
        return {
            "request_id": request_id,
            "model_name": request.model_name,
            "model_version": request.model_version,
            "to_stage": request.to_stage,
            "status": request.status,
            "urgency": request.urgency,
            "requested_by": request.requested_by,
            "request_timestamp": request.request_timestamp,
            "expiry_timestamp": request.expiry_timestamp,
            "approvals_received": approvals_count,
            "rejections_received": rejections_count,
            "required_approvals": request.min_approvals,
            "required_approvers": request.required_approvers,
            "final_decision": request.final_decision,
            "decision_timestamp": request.decision_timestamp,
            "time_remaining_hours": self._calculate_time_remaining(request),
            "approval_responses": request.approvals_received,
            "compliance_status": request.compliance_checks,
            "is_emergency": request.is_emergency
        }
    
    def list_pending_approvals(self, approver: str = None) -> List[Dict[str, Any]]:
        """List pending approval requests.
        
        Args:
            approver: Filter by approver (None for all)
            
        Returns:
            List of pending approval requests
        """
        pending_requests = []
        
        for request_id, request in self.pending_requests.items():
            # Filter by approver if specified
            if approver and approver not in request.required_approvers:
                continue
            
            # Check if approver already responded
            if approver:
                already_responded = any(
                    r["approver"] == approver for r in request.approvals_received
                )
                if already_responded:
                    continue
            
            status_info = self.get_approval_status(request_id)
            if status_info:
                pending_requests.append(status_info)
        
        # Sort by urgency and time remaining
        urgency_order = {"EMERGENCY": 0, "CRITICAL": 1, "HIGH": 2, "NORMAL": 3, "LOW": 4}
        pending_requests.sort(
            key=lambda x: (urgency_order.get(x["urgency"], 5), x["time_remaining_hours"])
        )
        
        return pending_requests
    
    def cancel_approval_request(self, request_id: str, reason: str = "") -> bool:
        """Cancel a pending approval request.
        
        Args:
            request_id: Request ID
            reason: Cancellation reason
            
        Returns:
            Success status
        """
        logger.info(f"Cancelling approval request: {request_id}")
        
        if request_id not in self.pending_requests:
            logger.error(f"Request {request_id} not found or already completed")
            return False
        
        approval_request = self.pending_requests[request_id]
        approval_request.status = ApprovalStatus.CANCELLED.value
        approval_request.decision_reason = reason
        approval_request.decision_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Update audit trail
        approval_request.audit_trail.append({
            "action": "request_cancelled",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": os.getenv("USER", "system"),
            "details": {"reason": reason}
        })
        
        # Move to completed requests
        self.completed_requests[request_id] = approval_request
        del self.pending_requests[request_id]
        
        # Send notification
        self.notification_manager.send_cancellation_notification(approval_request)
        
        logger.info(f"Approval request cancelled: {request_id}")
        return True
    
    def process_emergency_approval(self, model_name: str, model_version: str,
                                 to_stage: str, emergency_reason: str,
                                 emergency_approver: str, justification: str) -> str:
        """Process emergency approval request.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            to_stage: Target stage
            emergency_reason: Emergency reason code
            emergency_approver: Emergency approver
            justification: Detailed justification
            
        Returns:
            Request ID
        """
        logger.info(f"Processing emergency approval for {model_name} v{model_version}")
        
        # Validate emergency reason
        valid_reasons = self.config["approval_policies"]["emergency_deployment"]["valid_reasons"]
        if emergency_reason not in valid_reasons:
            raise ValueError(f"Invalid emergency reason. Valid reasons: {valid_reasons}")
        
        # Validate emergency approver
        emergency_approvers = self.config["approval_policies"]["emergency_deployment"]["approvers"]
        if emergency_approver not in emergency_approvers:
            raise ValueError(f"User {emergency_approver} not authorized for emergency approvals")
        
        # Create emergency approval request
        request_id = self.create_approval_request(
            model_name=model_name,
            model_version=model_version,
            to_stage=to_stage,
            approval_type=ApprovalType.EMERGENCY_DEPLOYMENT,
            reason=emergency_reason,
            description=justification,
            urgency=ApprovalUrgency.EMERGENCY,
            is_emergency=True,
            emergency_justification=justification
        )
        
        # Auto-approve with emergency approver
        approval_request = self.pending_requests[request_id]
        approval_request.emergency_approver = emergency_approver
        approval_request.status = ApprovalStatus.EMERGENCY_APPROVED.value
        approval_request.final_decision = "APPROVED"
        approval_request.decision_timestamp = datetime.now(timezone.utc).isoformat()
        approval_request.decision_reason = f"Emergency approval by {emergency_approver}"
        
        # Add emergency approval response
        emergency_response = ApprovalResponse(
            response_id=str(uuid.uuid4()),
            request_id=request_id,
            approver=emergency_approver,
            decision="APPROVE",
            timestamp=datetime.now(timezone.utc).isoformat(),
            comments=f"Emergency approval: {justification}",
            conditions=["emergency_monitoring_required"],
            digital_signature=self.signature_manager.create_emergency_signature(
                emergency_approver, justification, approval_request
            )
        )
        
        approval_request.approvals_received.append(asdict(emergency_response))
        
        # Update audit trail
        approval_request.audit_trail.append({
            "action": "emergency_approval_granted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": emergency_approver,
            "details": {
                "emergency_reason": emergency_reason,
                "justification": justification,
                "conditions": ["emergency_monitoring_required"]
            }
        })
        
        # Move to completed requests
        self.completed_requests[request_id] = approval_request
        del self.pending_requests[request_id]
        
        # Send emergency notifications
        self.notification_manager.send_emergency_approval_notification(approval_request)
        
        # Log to MLflow
        self._log_emergency_approval_to_mlflow(approval_request)
        
        logger.info(f"Emergency approval granted: {request_id}")
        return request_id
    
    def check_expired_requests(self) -> int:
        """Check for and process expired approval requests.
        
        Returns:
            Number of expired requests processed
        """
        expired_count = 0
        current_time = datetime.now(timezone.utc)
        
        expired_requests = []
        for request_id, request in self.pending_requests.items():
            expiry_time = datetime.fromisoformat(request.expiry_timestamp)
            if current_time > expiry_time:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            self._finalize_request(request_id, "EXPIRED", "Request expired without sufficient approvals")
            expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Processed {expired_count} expired approval requests")
        
        return expired_count
    
    def get_approval_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get approval workflow metrics.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Approval metrics
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Collect metrics from completed requests
        total_requests = 0
        approved_requests = 0
        rejected_requests = 0
        expired_requests = 0
        emergency_requests = 0
        
        approval_times = []
        urgency_counts = {"LOW": 0, "NORMAL": 0, "HIGH": 0, "CRITICAL": 0, "EMERGENCY": 0}
        
        for request in self.completed_requests.values():
            request_time = datetime.fromisoformat(request.request_timestamp)
            if request_time < cutoff_time:
                continue
            
            total_requests += 1
            urgency_counts[request.urgency] += 1
            
            if request.status == ApprovalStatus.APPROVED.value:
                approved_requests += 1
            elif request.status == ApprovalStatus.REJECTED.value:
                rejected_requests += 1
            elif request.status == ApprovalStatus.EXPIRED.value:
                expired_requests += 1
            elif request.status == ApprovalStatus.EMERGENCY_APPROVED.value:
                emergency_requests += 1
                approved_requests += 1
            
            # Calculate approval time
            if request.decision_timestamp:
                decision_time = datetime.fromisoformat(request.decision_timestamp)
                approval_time = (decision_time - request_time).total_seconds() / 3600  # hours
                approval_times.append(approval_time)
        
        # Calculate average approval time
        avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else 0
        
        return {
            "period_days": days,
            "total_requests": total_requests,
            "approved_requests": approved_requests,
            "rejected_requests": rejected_requests,
            "expired_requests": expired_requests,
            "emergency_requests": emergency_requests,
            "pending_requests": len(self.pending_requests),
            "approval_rate": (approved_requests / total_requests * 100) if total_requests > 0 else 0,
            "average_approval_time_hours": avg_approval_time,
            "urgency_distribution": urgency_counts,
            "metrics_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Helper methods
    def _get_approval_policy(self, approval_type: ApprovalType, stage: str, is_emergency: bool) -> Dict[str, Any]:
        """Get approval policy for given parameters."""
        if is_emergency:
            return self.config["approval_policies"]["emergency_deployment"]
        
        if approval_type == ApprovalType.STAGE_TRANSITION:
            return self.config["approval_policies"]["stage_transition"].get(stage.lower(), {
                "required": False, "approvers": [], "min_approvals": 0, "timeout_hours": 24
            })
        
        return {"required": False, "approvers": [], "min_approvals": 0, "timeout_hours": 24}
    
    def _check_auto_approval_conditions(self, request: ApprovalRequest) -> bool:
        """Check if request meets auto-approval conditions."""
        auto_rules = self.config.get("auto_approval_rules", {})
        if not auto_rules.get("enabled", False):
            return False
        
        conditions = auto_rules.get("conditions", {})
        
        # Check trusted users
        if request.requested_by in conditions.get("trusted_users", []):
            return True
        
        # Check non-production stages
        if request.to_stage.lower() in conditions.get("non_production_stages", []):
            return True
        
        # Check performance improvement
        performance_threshold = conditions.get("performance_improvement_threshold", 0.05)
        if self._check_performance_improvement(request, performance_threshold):
            return True
        
        return False
    
    def _check_performance_improvement(self, request: ApprovalRequest, threshold: float) -> bool:
        """Check if model shows performance improvement."""
        # Simplified implementation - would compare with current production model
        accuracy = request.performance_metrics.get("accuracy", 0)
        return accuracy > 0.85  # Simplified threshold
    
    def _auto_approve_request(self, model_name: str, model_version: str, to_stage: str, 
                            reason: str, request_id: str = None) -> str:
        """Auto-approve a request."""
        if not request_id:
            request_id = str(uuid.uuid4())
        
        logger.info(f"Auto-approving request: {request_id}")
        
        # Create minimal auto-approved request record
        auto_approved_request = ApprovalRequest(
            request_id=request_id,
            approval_type=ApprovalType.STAGE_TRANSITION.value,
            model_name=model_name,
            model_version=model_version,
            from_stage=None,
            to_stage=to_stage,
            requested_by=os.getenv("USER", "system"),
            request_timestamp=datetime.now(timezone.utc).isoformat(),
            expiry_timestamp=datetime.now(timezone.utc).isoformat(),
            urgency=ApprovalUrgency.NORMAL.value,
            reason=reason,
            description="Auto-approved based on policy",
            
            required_approvers=[],
            min_approvals=0,
            approvals_received=[],
            
            status=ApprovalStatus.AUTO_APPROVED.value,
            final_decision="APPROVED",
            decision_timestamp=datetime.now(timezone.utc).isoformat(),
            decision_reason="Auto-approved based on policy",
            
            validation_results={},
            performance_metrics={},
            impact_assessment={},
            risk_assessment={},
            
            compliance_checks={},
            audit_trail=[{
                "action": "auto_approved",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": "system",
                "details": {"reason": "Meets auto-approval criteria"}
            }],
            
            is_emergency=False,
            emergency_justification=None,
            emergency_approver=None
        )
        
        self.completed_requests[request_id] = auto_approved_request
        return request_id
    
    def _evaluate_approval_decision(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Evaluate if a final decision can be made."""
        approvals = [r for r in request.approvals_received if r["decision"] == "APPROVE"]
        rejections = [r for r in request.approvals_received if r["decision"] == "REJECT"]
        
        # Check for immediate rejection
        if len(rejections) > 0:
            return {
                "final_decision": "REJECTED",
                "reason": f"Request rejected by {rejections[0]['approver']}: {rejections[0]['comments']}"
            }
        
        # Check for sufficient approvals
        if len(approvals) >= request.min_approvals:
            return {
                "final_decision": "APPROVED",
                "reason": f"Received {len(approvals)} approvals (required: {request.min_approvals})"
            }
        
        return {"final_decision": None, "reason": "Insufficient approvals"}
    
    def _finalize_request(self, request_id: str, final_decision: str, reason: str):
        """Finalize an approval request."""
        if request_id not in self.pending_requests:
            return
        
        approval_request = self.pending_requests[request_id]
        approval_request.final_decision = final_decision
        approval_request.decision_reason = reason
        approval_request.decision_timestamp = datetime.now(timezone.utc).isoformat()
        approval_request.status = final_decision
        
        # Update audit trail
        approval_request.audit_trail.append({
            "action": "request_finalized",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": "system",
            "details": {"final_decision": final_decision, "reason": reason}
        })
        
        # Move to completed requests
        self.completed_requests[request_id] = approval_request
        del self.pending_requests[request_id]
        
        # Send final notification
        self.notification_manager.send_final_decision_notification(approval_request)
        
        # Log to MLflow
        self._log_final_decision_to_mlflow(approval_request)
    
    def _calculate_time_remaining(self, request: ApprovalRequest) -> float:
        """Calculate time remaining for approval request in hours."""
        current_time = datetime.now(timezone.utc)
        expiry_time = datetime.fromisoformat(request.expiry_timestamp)
        
        time_diff = expiry_time - current_time
        return max(0, time_diff.total_seconds() / 3600)
    
    def _get_validation_results(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get model validation results."""
        # Simplified implementation - would integrate with model validation system
        return {"validation_passed": True, "validation_score": 0.95}
    
    def _get_performance_metrics(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            mv = self.mlflow_client.get_model_version(model_name, model_version)
            run = self.mlflow_client.get_run(mv.run_id)
            return run.data.metrics
        except:
            return {}
    
    def _assess_impact(self, model_name: str, model_version: str, to_stage: str) -> Dict[str, Any]:
        """Assess deployment impact."""
        return {
            "user_impact": "medium",
            "business_impact": "low",
            "technical_impact": "low",
            "compliance_impact": "none"
        }
    
    def _assess_risk(self, model_name: str, model_version: str, to_stage: str) -> Dict[str, Any]:
        """Assess deployment risk."""
        return {
            "technical_risk": "low",
            "business_risk": "low",
            "security_risk": "low",
            "compliance_risk": "none",
            "overall_risk": "low"
        }
    
    def _log_approval_request_to_mlflow(self, request: ApprovalRequest):
        """Log approval request to MLflow."""
        try:
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("model_approval_workflows").experiment_id):
                mlflow.log_param("request_id", request.request_id)
                mlflow.log_param("model_name", request.model_name)
                mlflow.log_param("model_version", request.model_version)
                mlflow.log_param("to_stage", request.to_stage)
                mlflow.log_param("approval_type", request.approval_type)
                mlflow.log_param("urgency", request.urgency)
                mlflow.log_param("is_emergency", request.is_emergency)
                mlflow.log_param("required_approvals", request.min_approvals)
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
    
    def _log_approval_response_to_mlflow(self, request: ApprovalRequest, response: ApprovalResponse):
        """Log approval response to MLflow."""
        # Implementation would log response details
        pass
    
    def _log_final_decision_to_mlflow(self, request: ApprovalRequest):
        """Log final decision to MLflow."""
        # Implementation would log final decision and metrics
        pass
    
    def _log_emergency_approval_to_mlflow(self, request: ApprovalRequest):
        """Log emergency approval to MLflow."""
        # Implementation would log emergency approval details
        pass


class NotificationManager:
    """Manages approval workflow notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def send_approval_request_notification(self, request: ApprovalRequest):
        """Send approval request notification."""
        logger.info(f"Sending approval request notification for {request.request_id}")
        # Implementation would send actual notifications
    
    def send_approval_response_notification(self, request: ApprovalRequest, response: ApprovalResponse):
        """Send approval response notification."""
        logger.info(f"Sending approval response notification for {request.request_id}")
        # Implementation would send actual notifications
    
    def send_final_decision_notification(self, request: ApprovalRequest):
        """Send final decision notification."""
        logger.info(f"Sending final decision notification for {request.request_id}")
        # Implementation would send actual notifications
    
    def send_emergency_approval_notification(self, request: ApprovalRequest):
        """Send emergency approval notification."""
        logger.info(f"Sending emergency approval notification for {request.request_id}")
        # Implementation would send actual notifications
    
    def send_cancellation_notification(self, request: ApprovalRequest):
        """Send cancellation notification."""
        logger.info(f"Sending cancellation notification for {request.request_id}")
        # Implementation would send actual notifications


class DigitalSignatureManager:
    """Manages digital signatures for compliance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signatures_enabled = config.get("compliance_config", {}).get("electronic_signatures", False)
    
    def create_signature(self, approver: str, decision: str, comments: str, 
                        request: ApprovalRequest) -> Optional[str]:
        """Create digital signature for approval."""
        if not self.signatures_enabled:
            return None
        
        # Create signature content
        signature_content = f"{approver}:{decision}:{comments}:{request.request_id}:{datetime.now(timezone.utc).isoformat()}"
        
        # Create hash signature (simplified - would use proper digital signatures)
        signature = hashlib.sha256(signature_content.encode()).hexdigest()
        
        logger.info(f"Created digital signature for {approver}")
        return signature
    
    def create_emergency_signature(self, approver: str, justification: str, 
                                 request: ApprovalRequest) -> Optional[str]:
        """Create emergency approval signature."""
        if not self.signatures_enabled:
            return None
        
        signature_content = f"EMERGENCY:{approver}:{justification}:{request.request_id}:{datetime.now(timezone.utc).isoformat()}"
        signature = hashlib.sha256(signature_content.encode()).hexdigest()
        
        logger.info(f"Created emergency signature for {approver}")
        return signature


class ComplianceValidator:
    """Validates compliance requirements."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def validate_compliance(self, model_name: str, model_version: str, to_stage: str) -> Dict[str, Any]:
        """Validate compliance requirements."""
        compliance_config = self.config.get("compliance_config", {})
        
        checks = {
            "fda_21_cfr_part_11": self._check_fda_compliance(),
            "audit_trail_complete": self._check_audit_trail(),
            "validation_documentation": self._check_validation_docs(),
            "change_control": self._check_change_control(),
            "electronic_signatures": compliance_config.get("electronic_signatures", False)
        }
        
        all_passed = all(checks.values())
        
        return {
            "compliance_passed": all_passed,
            "compliance_checks": checks,
            "compliance_score": sum(checks.values()) / len(checks),
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_fda_compliance(self) -> bool:
        """Check FDA 21 CFR Part 11 compliance."""
        # Simplified implementation
        return True
    
    def _check_audit_trail(self) -> bool:
        """Check audit trail completeness."""
        # Simplified implementation  
        return True
    
    def _check_validation_docs(self) -> bool:
        """Check validation documentation."""
        # Simplified implementation
        return True
    
    def _check_change_control(self) -> bool:
        """Check change control process."""
        # Simplified implementation
        return True


class EmergencyApprovalManager:
    """Manages emergency approval procedures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.emergency_log = []
    
    def validate_emergency_request(self, reason: str, approver: str, justification: str) -> bool:
        """Validate emergency approval request."""
        # Check if reason is valid
        valid_reasons = self.config["approval_policies"]["emergency_deployment"]["valid_reasons"]
        if reason not in valid_reasons:
            return False
        
        # Check if approver is authorized
        emergency_approvers = self.config["approval_policies"]["emergency_deployment"]["approvers"]
        if approver not in emergency_approvers:
            return False
        
        # Log emergency request
        self.emergency_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "approver": approver,
            "justification": justification
        })
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model Approval Workflows")
    parser.add_argument("--action", required=True,
                       choices=["create", "approve", "reject", "status", "list", "cancel", "emergency", "metrics"],
                       help="Action to perform")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--model-version", help="Model version")
    parser.add_argument("--to-stage", help="Target stage")
    parser.add_argument("--from-stage", help="Source stage")
    parser.add_argument("--request-id", help="Approval request ID")
    parser.add_argument("--approver", help="Approver name")
    parser.add_argument("--decision", choices=["APPROVE", "REJECT", "ABSTAIN"], help="Approval decision")
    parser.add_argument("--comments", help="Approval comments")
    parser.add_argument("--reason", help="Request reason")
    parser.add_argument("--urgency", choices=["LOW", "NORMAL", "HIGH", "CRITICAL", "EMERGENCY"], 
                       default="NORMAL", help="Request urgency")
    parser.add_argument("--emergency-reason", help="Emergency reason code")
    parser.add_argument("--justification", help="Emergency justification")
    parser.add_argument("--config", default="configs/approval_workflows_config.yaml", help="Configuration file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize workflow manager
    manager = ApprovalWorkflowManager(args.config)
    
    try:
        if args.action == "create":
            if not all([args.model_name, args.model_version, args.to_stage]):
                print("Error: model-name, model-version, and to-stage are required")
                sys.exit(1)
            
            request_id = manager.create_approval_request(
                model_name=args.model_name,
                model_version=args.model_version,
                to_stage=args.to_stage,
                from_stage=args.from_stage,
                reason=args.reason or "",
                urgency=ApprovalUrgency(args.urgency)
            )
            print(f"Approval request created: {request_id}")
        
        elif args.action == "approve":
            if not all([args.request_id, args.approver]):
                print("Error: request-id and approver are required for approval")
                sys.exit(1)
            
            success = manager.submit_approval_response(
                args.request_id, args.approver, "APPROVE", args.comments or ""
            )
            print(f"Approval {'submitted' if success else 'failed'}")
        
        elif args.action == "reject":
            if not all([args.request_id, args.approver]):
                print("Error: request-id and approver are required for rejection")
                sys.exit(1)
            
            success = manager.submit_approval_response(
                args.request_id, args.approver, "REJECT", args.comments or ""
            )
            print(f"Rejection {'submitted' if success else 'failed'}")
        
        elif args.action == "status":
            if not args.request_id:
                print("Error: request-id is required for status")
                sys.exit(1)
            
            status = manager.get_approval_status(args.request_id)
            if status:
                print(json.dumps(status, indent=2))
            else:
                print(f"Request {args.request_id} not found")
        
        elif args.action == "list":
            pending = manager.list_pending_approvals(args.approver)
            print(f"Found {len(pending)} pending approval requests")
            for request in pending:
                print(f"  {request['request_id']} - {request['model_name']} v{request['model_version']} -> {request['to_stage']}")
                print(f"    Urgency: {request['urgency']}, Time remaining: {request['time_remaining_hours']:.1f}h")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(pending, f, indent=2)
        
        elif args.action == "cancel":
            if not args.request_id:
                print("Error: request-id is required for cancellation")
                sys.exit(1)
            
            success = manager.cancel_approval_request(args.request_id, args.reason or "")
            print(f"Cancellation {'succeeded' if success else 'failed'}")
        
        elif args.action == "emergency":
            if not all([args.model_name, args.model_version, args.to_stage, 
                       args.emergency_reason, args.approver, args.justification]):
                print("Error: model-name, model-version, to-stage, emergency-reason, approver, and justification are required")
                sys.exit(1)
            
            request_id = manager.process_emergency_approval(
                args.model_name, args.model_version, args.to_stage,
                args.emergency_reason, args.approver, args.justification
            )
            print(f"Emergency approval granted: {request_id}")
        
        elif args.action == "metrics":
            metrics = manager.get_approval_metrics()
            print(json.dumps(metrics, indent=2))
        
        # Check for expired requests
        expired_count = manager.check_expired_requests()
        if expired_count > 0:
            print(f"Note: {expired_count} requests expired and were automatically processed")
    
    except Exception as e:
        logger.error(f"Approval workflow operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()