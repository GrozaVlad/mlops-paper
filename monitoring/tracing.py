#!/usr/bin/env python3
"""
Distributed Tracing Implementation for DrugBAN MLOps Pipeline
Uses OpenTelemetry and Jaeger for comprehensive request flow tracking
"""

import time
import logging
import os
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import uuid

# OpenTelemetry imports
from opentelemetry import trace, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.semantic_conventions.resource import ResourceAttributes
from opentelemetry import context as context_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugBANTracer:
    """
    Main tracing class for DrugBAN MLOps pipeline
    """
    
    def __init__(self, service_name: str = "drugban", 
                 jaeger_endpoint: str = "http://jaeger:14268/api/traces",
                 environment: str = "production"):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.environment = environment
        self.tracer = None
        self._setup_tracing()
        
    def _setup_tracing(self):
        """Initialize OpenTelemetry tracing"""
        
        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment,
            "service.instance.id": str(uuid.uuid4())
        })
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
            collector_endpoint=self.jaeger_endpoint,
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Optional: Add console exporter for debugging
        if os.getenv("OTEL_DEBUG", "false").lower() == "true":
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)
        
        # Set up propagators
        set_global_textmap(B3MultiFormat())
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument common libraries
        self._setup_auto_instrumentation()
        
        logger.info(f"Tracing initialized for service: {self.service_name}")
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries"""
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument Redis (if available)
            RedisInstrumentor().instrument()
            
            # Instrument SQLAlchemy (if available)
            SQLAlchemyInstrumentor().instrument()
            
            logger.info("Auto-instrumentation setup completed")
        except Exception as e:
            logger.warning(f"Some auto-instrumentation failed: {e}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application"""
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation completed")
    
    @contextmanager
    def span(self, name: str, **attributes):
        """Create a new span with context manager"""
        with self.tracer.start_as_current_span(name) as span:
            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span
    
    def trace_function(self, span_name: Optional[str] = None, 
                      include_args: bool = False,
                      include_result: bool = False):
        """Decorator to trace function calls"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_as_current_span(name) as span:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add arguments if requested
                    if include_args:
                        for i, arg in enumerate(args):
                            span.set_attribute(f"function.arg.{i}", str(arg)[:100])
                        for key, value in kwargs.items():
                            span.set_attribute(f"function.kwarg.{key}", str(value)[:100])
                    
                    try:
                        # Execute function
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        
                        # Add execution metrics
                        span.set_attribute("function.execution_time", execution_time)
                        span.set_attribute("function.status", "success")
                        
                        # Add result if requested
                        if include_result and result is not None:
                            span.set_attribute("function.result", str(result)[:100])
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.set_attribute("function.status", "error")
                        span.set_attribute("function.error", str(e))
                        span.record_exception(e)
                        raise
                        
            return wrapper
        return decorator

# Global tracer instance
_tracer_instance: Optional[DrugBANTracer] = None

def get_tracer(service_name: str = "drugban") -> DrugBANTracer:
    """Get or create global tracer instance"""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = DrugBANTracer(service_name=service_name)
    return _tracer_instance

def trace_prediction_pipeline():
    """Decorator specifically for prediction pipeline tracing"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.span("prediction_pipeline") as span:
                # Extract prediction details
                if args and hasattr(args[0], 'drug_id'):
                    span.set_attribute("prediction.drug_id", args[0].drug_id)
                if args and hasattr(args[0], 'target_id'):
                    span.set_attribute("prediction.target_id", args[0].target_id)
                
                # Add baggage for downstream services
                baggage.set_baggage("prediction.session", str(uuid.uuid4()))
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add result metrics
                    if hasattr(result, 'prediction'):
                        span.set_attribute("prediction.score", result.prediction)
                    if hasattr(result, 'confidence'):
                        span.set_attribute("prediction.confidence", result.confidence)
                    
                    span.set_attribute("prediction.status", "success")
                    return result
                    
                except Exception as e:
                    span.set_attribute("prediction.status", "failed")
                    span.set_attribute("prediction.error", str(e))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator

def trace_feature_extraction():
    """Decorator for feature extraction tracing"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.span("feature_extraction") as span:
                # Extract input details
                if args:
                    span.set_attribute("feature.input_type", type(args[0]).__name__)
                    if hasattr(args[0], '__len__'):
                        span.set_attribute("feature.input_size", len(args[0]))
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    extraction_time = time.time() - start_time
                    
                    # Add extraction metrics
                    span.set_attribute("feature.extraction_time", extraction_time)
                    span.set_attribute("feature.status", "success")
                    
                    if hasattr(result, 'shape'):
                        span.set_attribute("feature.output_shape", str(result.shape))
                    elif hasattr(result, '__len__'):
                        span.set_attribute("feature.output_size", len(result))
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("feature.status", "failed")
                    span.set_attribute("feature.error", str(e))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator

def trace_model_inference():
    """Decorator for model inference tracing"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.span("model_inference") as span:
                # Model metadata
                if hasattr(args[0], '__class__'):
                    span.set_attribute("model.type", args[0].__class__.__name__)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    inference_time = time.time() - start_time
                    
                    # Add inference metrics
                    span.set_attribute("model.inference_time", inference_time)
                    span.set_attribute("model.status", "success")
                    
                    # Add prediction statistics
                    if hasattr(result, 'shape'):
                        span.set_attribute("model.output_shape", str(result.shape))
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("model.status", "failed")
                    span.set_attribute("model.error", str(e))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator

def trace_drift_detection():
    """Decorator for drift detection tracing"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.span("drift_detection") as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    detection_time = time.time() - start_time
                    
                    # Add drift detection metrics
                    span.set_attribute("drift.detection_time", detection_time)
                    span.set_attribute("drift.status", "success")
                    
                    if hasattr(result, 'drift_detected'):
                        span.set_attribute("drift.detected", result.drift_detected)
                    if hasattr(result, 'drift_score'):
                        span.set_attribute("drift.score", result.drift_score)
                    if hasattr(result, 'drift_method'):
                        span.set_attribute("drift.method", result.drift_method)
                    
                    return result
                    
                except Exception as e:
                    span.set_attribute("drift.status", "failed")
                    span.set_attribute("drift.error", str(e))
                    span.record_exception(e)
                    raise
                    
        return wrapper
    return decorator

class TracingMiddleware:
    """Middleware for adding tracing to requests"""
    
    def __init__(self, app, tracer: DrugBANTracer):
        self.app = app
        self.tracer = tracer
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Create span for request
            with self.tracer.span("http_request") as span:
                # Add request details
                span.set_attribute("http.method", scope["method"])
                span.set_attribute("http.path", scope["path"])
                span.set_attribute("http.scheme", scope["scheme"])
                
                # Add query parameters
                if scope.get("query_string"):
                    span.set_attribute("http.query_string", scope["query_string"].decode())
                
                # Add headers (selective)
                headers = dict(scope.get("headers", []))
                if b"user-agent" in headers:
                    span.set_attribute("http.user_agent", headers[b"user-agent"].decode())
                if b"x-forwarded-for" in headers:
                    span.set_attribute("http.client_ip", headers[b"x-forwarded-for"].decode())
                
                # Process request
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)

def create_trace_context(operation_id: str, **attributes) -> Dict[str, Any]:
    """Create a trace context for manual span creation"""
    tracer = get_tracer()
    
    span = tracer.tracer.start_span(operation_id)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    
    return {
        "span": span,
        "context": trace.set_span_in_context(span)
    }

def end_trace_context(trace_context: Dict[str, Any], **final_attributes):
    """End a manually created trace context"""
    span = trace_context["span"]
    
    for key, value in final_attributes.items():
        span.set_attribute(key, value)
    
    span.end()

# Utility functions for common tracing patterns
def trace_batch_operation(operation_name: str, batch_size: int):
    """Create a span for batch operations"""
    tracer = get_tracer()
    
    @contextmanager
    def batch_span():
        with tracer.span(operation_name) as span:
            span.set_attribute("batch.size", batch_size)
            span.set_attribute("batch.operation", operation_name)
            yield span
    
    return batch_span()

def add_prediction_metrics(span, prediction_result):
    """Add prediction-specific metrics to a span"""
    if hasattr(prediction_result, 'prediction'):
        span.set_attribute("prediction.value", prediction_result.prediction)
    if hasattr(prediction_result, 'confidence'):
        span.set_attribute("prediction.confidence", prediction_result.confidence)
    if hasattr(prediction_result, 'processing_time'):
        span.set_attribute("prediction.processing_time", prediction_result.processing_time)

def add_error_metrics(span, error: Exception):
    """Add error metrics to a span"""
    span.set_attribute("error.type", type(error).__name__)
    span.set_attribute("error.message", str(error))
    span.record_exception(error)

# Integration helpers
def setup_tracing_for_fastapi(app, service_name: str = "drugban-api"):
    """Set up complete tracing for FastAPI application"""
    tracer = get_tracer(service_name)
    tracer.instrument_fastapi(app)
    
    # Add custom middleware
    app.add_middleware(TracingMiddleware, tracer=tracer)
    
    return tracer

def get_current_trace_id() -> Optional[str]:
    """Get current trace ID for correlation"""
    span = trace.get_current_span()
    if span and span.get_span_context().trace_id:
        return hex(span.get_span_context().trace_id)[2:]
    return None

def get_current_span_id() -> Optional[str]:
    """Get current span ID for correlation"""
    span = trace.get_current_span()
    if span and span.get_span_context().span_id:
        return hex(span.get_span_context().span_id)[2:]
    return None

if __name__ == "__main__":
    # Example usage
    tracer = DrugBANTracer()
    
    # Example traced function
    @tracer.trace_function("example_function", include_args=True, include_result=True)
    def example_function(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x + y
    
    # Example prediction pipeline
    @trace_prediction_pipeline()
    def predict_drug_target(drug_id: str, target_id: str):
        time.sleep(0.2)  # Simulate prediction
        return {
            "prediction": 0.85,
            "confidence": 0.92
        }
    
    # Run examples
    with tracer.span("example_operation", operation_type="demo"):
        result1 = example_function(5, 3)
        result2 = predict_drug_target("DRUG123", "TARGET456")
        
        print(f"Results: {result1}, {result2}")
        print(f"Trace ID: {get_current_trace_id()}")
    
    print("Tracing example completed")