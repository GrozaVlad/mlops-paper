#!/usr/bin/env python3
"""
FastAPI Application for DrugBAN Model Serving

This module provides a REST API for drug-target interaction prediction
with authentication, rate limiting, and comprehensive validation.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import torch
import numpy as np
import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import DrugTargetInteractionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_PATH = os.getenv("MODEL_PATH", "models/trained")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")

# Initialize FastAPI app
app = FastAPI(
    title="DrugBAN Prediction API",
    description="REST API for drug-target interaction prediction using DrugBAN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize rate limiter
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
except:
    logger.warning("Redis not available, using in-memory rate limiting")
    limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Global variables for model loading
model_instance = None
model_loaded = False

# Pydantic models for request/response validation
class DrugInfo(BaseModel):
    """Drug information model."""
    drug_id: str = Field(..., description="Unique drug identifier")
    smiles: str = Field(..., description="SMILES representation of the drug")
    drug_name: Optional[str] = Field(None, description="Drug name")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES string cannot be empty")
        # Basic SMILES validation - could be enhanced with RDKit
        if len(v) > 1000:
            raise ValueError("SMILES string too long")
        return v.strip()

class TargetInfo(BaseModel):
    """Target protein information model."""
    target_id: str = Field(..., description="Unique target identifier")
    target_name: str = Field(..., description="Target protein name")
    uniprot_id: Optional[str] = Field(None, description="UniProt identifier")
    organism: Optional[str] = Field("Homo sapiens", description="Target organism")
    target_class: Optional[str] = Field("unknown", description="Target class (GPCR, enzyme, etc.)")

class PredictionRequest(BaseModel):
    """Drug-target interaction prediction request."""
    drug: DrugInfo
    target: TargetInfo
    return_confidence: Optional[bool] = Field(True, description="Return prediction confidence")
    return_features: Optional[bool] = Field(False, description="Return extracted features")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    predictions: List[PredictionRequest] = Field(..., max_items=100, description="List of prediction requests (max 100)")

class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: float = Field(..., description="Interaction probability (0-1)")
    prediction_class: str = Field(..., description="Predicted class (agonist/antagonist)")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted features")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    results: List[PredictionResponse]
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    version: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and get current user."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    return token_data

# Model loading functions
async def load_model():
    """Load the trained model."""
    global model_instance, model_loaded
    
    try:
        logger.info("Loading DrugBAN model...")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load latest model from MLflow
        try:
            # Try to load from model registry
            model_uri = "models:/DrugTargetInteractionModel/latest"
            model_instance = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from MLflow registry: {model_uri}")
        except:
            # Fallback to latest run
            experiment = mlflow.get_experiment_by_name("drug_repurposing_training")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    latest_run = runs.iloc[0]
                    model_uri = f"runs:/{latest_run.run_id}/model"
                    model_instance = mlflow.pytorch.load_model(model_uri)
                    logger.info(f"Loaded model from latest run: {model_uri}")
                else:
                    raise Exception("No training runs found")
            else:
                raise Exception("Training experiment not found")
        
        # Set model to evaluation mode
        model_instance.eval()
        model_loaded = True
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        raise

def predict_interaction(drug_features: np.ndarray, target_features: np.ndarray) -> Dict[str, Any]:
    """Make prediction using the loaded model."""
    if not model_loaded or model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Convert to tensors
        drug_tensor = torch.FloatTensor(drug_features).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_features).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model_instance(drug_tensor, target_tensor)
            probability = prediction.item()
            
        # Determine class
        prediction_class = "agonist" if probability > 0.5 else "antagonist"
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "prediction": probability,
            "prediction_class": prediction_class,
            "confidence": confidence,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

def extract_features(drug_info: DrugInfo, target_info: TargetInfo) -> tuple:
    """Extract features from drug and target information."""
    # This is a simplified feature extraction
    # In production, you would use the same feature extraction pipeline
    # as used during training
    
    # Placeholder feature extraction
    # Drug features (simplified - would use actual molecular fingerprints)
    drug_features = np.random.random(2079)  # Combined morgan + descriptors
    
    # Target features (simplified - would use actual protein features)
    target_features = np.random.random(8)  # Target encoding features
    
    return drug_features, target_features

# API Routes

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting DrugBAN Prediction API...")
    await load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.post("/auth/token")
async def login_for_access_token(request: Request):
    """Generate access token for API authentication."""
    # Simplified authentication - implement proper user validation
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "api_user"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_drug_target_interaction(
    request: Request,
    prediction_request: PredictionRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """Predict drug-target interaction."""
    try:
        start_time = datetime.now()
        
        # Extract features
        drug_features, target_features = extract_features(
            prediction_request.drug, 
            prediction_request.target
        )
        
        # Make prediction
        result = predict_interaction(drug_features, target_features)
        
        # Prepare response
        response = PredictionResponse(
            prediction=result["prediction"],
            prediction_class=result["prediction_class"],
            processing_time_ms=result["processing_time_ms"]
        )
        
        if prediction_request.return_confidence:
            response.confidence = result["confidence"]
            
        if prediction_request.return_features:
            response.features = {
                "drug_features_shape": drug_features.shape,
                "target_features_shape": target_features.shape
            }
        
        # Log prediction
        logger.info(f"Prediction made for {prediction_request.drug.drug_id} -> {prediction_request.target.target_id}: {result['prediction']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("2/minute")
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    """Batch prediction endpoint."""
    start_time = datetime.now()
    results = []
    successful = 0
    failed = 0
    
    for pred_request in batch_request.predictions:
        try:
            # Extract features
            drug_features, target_features = extract_features(
                pred_request.drug, 
                pred_request.target
            )
            
            # Make prediction
            result = predict_interaction(drug_features, target_features)
            
            response = PredictionResponse(
                prediction=result["prediction"],
                prediction_class=result["prediction_class"],
                confidence=result["confidence"] if pred_request.return_confidence else None,
                processing_time_ms=result["processing_time_ms"]
            )
            
            results.append(response)
            successful += 1
            
        except Exception as e:
            logger.error(f"Batch prediction error for {pred_request.drug.drug_id}: {e}")
            failed += 1
            
            # Add error response
            error_response = PredictionResponse(
                prediction=0.0,
                prediction_class="error",
                processing_time_ms=0.0
            )
            results.append(error_response)
    
    total_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return BatchPredictionResponse(
        results=results,
        total_predictions=len(batch_request.predictions),
        successful_predictions=successful,
        failed_predictions=failed,
        total_processing_time_ms=total_time
    )

@app.get("/model/info")
async def model_info(current_user: TokenData = Depends(get_current_user)):
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": model_loaded,
        "model_type": "DrugTargetInteractionModel",
        "version": "1.0.0",
        "framework": "PyTorch",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get API metrics (for Prometheus scraping)."""
    # This would typically integrate with Prometheus metrics
    return {
        "predictions_total": 0,  # Would be tracked
        "predictions_per_second": 0.0,
        "model_loaded": model_loaded,
        "uptime_seconds": 0,  # Would be calculated
        "memory_usage_mb": 0,  # Would be monitored
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )