#!/usr/bin/env python3
"""
Log Analysis and Debugging Tools for DrugBAN MLOps Pipeline
Provides comprehensive log analysis, pattern detection, and optimization insights
"""

import re
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import asyncio
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Structure for parsed log entries"""
    timestamp: datetime
    level: str
    service: str
    component: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    latency: Optional[float] = None
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    drug_id: Optional[str] = None
    target_id: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisResult:
    """Result of log analysis"""
    analysis_type: str
    timestamp: datetime
    summary: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    visualizations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LogPatternMatcher:
    """Pattern matching for different log types"""
    
    def __init__(self):
        self.patterns = {
            'api_request': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+'
                r'(?P<level>\w+)\s+'
                r'(?P<service>\w+)\s+-\s+'
                r'(?P<method>\w+)\s+(?P<path>/\S+)\s+'
                r'(?P<status_code>\d+)\s+'
                r'(?P<latency>\d+\.\d+)ms'
            ),
            'prediction_log': re.compile(
                r'prediction_id=(?P<prediction_id>\S+)\s+'
                r'drug_id=(?P<drug_id>\S+)\s+'
                r'target_id=(?P<target_id>\S+)\s+'
                r'confidence=(?P<confidence>\d+\.\d+)\s+'
                r'latency=(?P<latency>\d+\.\d+)'
            ),
            'error_log': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+'
                r'ERROR\s+(?P<service>\w+)\s+-\s+'
                r'(?P<error_type>\w+):\s*(?P<error_message>.*)'
            ),
            'drift_detection': re.compile(
                r'drift_type=(?P<drift_type>\w+)\s+'
                r'drift_score=(?P<drift_score>\d+\.\d+)\s+'
                r'threshold=(?P<threshold>\d+\.\d+)\s+'
                r'drift_detected=(?P<drift_detected>true|false)'
            ),
            'performance_metric': re.compile(
                r'metric=(?P<metric_name>\w+)\s+'
                r'value=(?P<metric_value>\d+\.\d+)\s+'
                r'window_size=(?P<window_size>\d+)'
            )
        }
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry"""
        
        # Try each pattern
        for pattern_name, pattern in self.patterns.items():
            match = pattern.search(line)
            if match:
                return self._create_log_entry(pattern_name, match, line)
        
        # Fallback generic parsing
        return self._parse_generic_log(line)
    
    def _create_log_entry(self, pattern_name: str, match: re.Match, line: str) -> LogEntry:
        """Create LogEntry from regex match"""
        data = match.groupdict()
        
        # Parse timestamp
        timestamp = datetime.now()
        if 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                pass
        
        return LogEntry(
            timestamp=timestamp,
            level=data.get('level', 'INFO'),
            service=data.get('service', 'unknown'),
            component=pattern_name,
            message=line,
            latency=float(data['latency']) if 'latency' in data else None,
            status_code=int(data['status_code']) if 'status_code' in data else None,
            error_type=data.get('error_type'),
            drug_id=data.get('drug_id'),
            target_id=data.get('target_id'),
            confidence=float(data['confidence']) if 'confidence' in data else None
        )
    
    def _parse_generic_log(self, line: str) -> LogEntry:
        """Parse generic log format"""
        parts = line.split()
        
        timestamp = datetime.now()
        level = 'INFO'
        service = 'unknown'
        
        if len(parts) > 0:
            try:
                timestamp = datetime.fromisoformat(parts[0].replace('Z', '+00:00'))
            except ValueError:
                pass
        
        if len(parts) > 1 and parts[1] in ['DEBUG', 'INFO', 'WARN', 'ERROR']:
            level = parts[1]
        
        if len(parts) > 2:
            service = parts[2]
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            service=service,
            component='generic',
            message=line
        )

class LogAnalyzer:
    """Main log analysis engine"""
    
    def __init__(self, elasticsearch_host: str = "elasticsearch:9200"):
        self.es_host = elasticsearch_host
        self.es_client = None
        self.pattern_matcher = LogPatternMatcher()
        
        try:
            self.es_client = Elasticsearch([elasticsearch_host])
        except Exception as e:
            logger.warning(f"Could not connect to Elasticsearch: {e}")
    
    def analyze_logs(self, start_time: datetime, end_time: datetime, 
                    services: Optional[List[str]] = None) -> List[AnalysisResult]:
        """Perform comprehensive log analysis"""
        
        # Get logs from Elasticsearch or files
        logs = self._fetch_logs(start_time, end_time, services)
        
        if not logs:
            logger.warning("No logs found for analysis")
            return []
        
        # Perform different types of analysis
        analyses = []
        
        # Error analysis
        error_analysis = self._analyze_errors(logs)
        if error_analysis:
            analyses.append(error_analysis)
        
        # Performance analysis
        performance_analysis = self._analyze_performance(logs)
        if performance_analysis:
            analyses.append(performance_analysis)
        
        # Prediction quality analysis
        prediction_analysis = self._analyze_predictions(logs)
        if prediction_analysis:
            analyses.append(prediction_analysis)
        
        # Traffic pattern analysis
        traffic_analysis = self._analyze_traffic_patterns(logs)
        if traffic_analysis:
            analyses.append(traffic_analysis)
        
        # Drift analysis
        drift_analysis = self._analyze_drift_patterns(logs)
        if drift_analysis:
            analyses.append(drift_analysis)
        
        return analyses
    
    def _fetch_logs(self, start_time: datetime, end_time: datetime, 
                   services: Optional[List[str]] = None) -> List[LogEntry]:
        """Fetch logs from Elasticsearch or local files"""
        
        logs = []
        
        if self.es_client:
            logs = self._fetch_from_elasticsearch(start_time, end_time, services)
        else:
            logs = self._fetch_from_files(start_time, end_time, services)
        
        return logs
    
    def _fetch_from_elasticsearch(self, start_time: datetime, end_time: datetime,
                                 services: Optional[List[str]] = None) -> List[LogEntry]:
        """Fetch logs from Elasticsearch"""
        
        try:
            # Build query
            query = {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat()
                                }
                            }
                        }
                    ]
                }
            }
            
            if services:
                query["bool"]["must"].append({
                    "terms": {
                        "service_type": services
                    }
                })
            
            # Search logs
            response = self.es_client.search(
                index="drugban-logs-*",
                body={
                    "query": query,
                    "size": 10000,  # Adjust as needed
                    "sort": [{"@timestamp": {"order": "asc"}}]
                }
            )
            
            # Parse results
            logs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                log_entry = LogEntry(
                    timestamp=datetime.fromisoformat(source["@timestamp"]),
                    level=source.get("log_level", "INFO"),
                    service=source.get("service_type", "unknown"),
                    component=source.get("component", "unknown"),
                    message=source.get("message", ""),
                    trace_id=source.get("trace_id"),
                    span_id=source.get("span_id"),
                    latency=source.get("latency") or source.get("response_latency"),
                    status_code=source.get("status_code"),
                    error_type=source.get("error_type"),
                    drug_id=source.get("drug_id"),
                    target_id=source.get("target_id"),
                    confidence=source.get("confidence")
                )
                logs.append(log_entry)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error fetching from Elasticsearch: {e}")
            return []
    
    def _fetch_from_files(self, start_time: datetime, end_time: datetime,
                         services: Optional[List[str]] = None) -> List[LogEntry]:
        """Fetch logs from local files (fallback)"""
        
        # This would read from local log files
        # For now, return empty list as placeholder
        logger.info("Fetching from local files not implemented")
        return []
    
    def _analyze_errors(self, logs: List[LogEntry]) -> Optional[AnalysisResult]:
        """Analyze error patterns and trends"""
        
        error_logs = [log for log in logs if log.level == 'ERROR' or log.error_type]
        
        if not error_logs:
            return None
        
        # Error statistics
        total_errors = len(error_logs)
        error_types = Counter([log.error_type for log in error_logs if log.error_type])
        services_with_errors = Counter([log.service for log in error_logs])
        
        # Time-based error analysis
        error_timeline = defaultdict(int)
        for log in error_logs:
            hour = log.timestamp.replace(minute=0, second=0, microsecond=0)
            error_timeline[hour] += 1
        
        # Error rate calculation
        total_logs = len(logs)
        error_rate = (total_errors / total_logs) * 100 if total_logs > 0 else 0
        
        # Insights generation
        insights = [
            f"Total errors: {total_errors} ({error_rate:.2f}% error rate)",
            f"Most common error type: {error_types.most_common(1)[0] if error_types else 'None'}",
            f"Service with most errors: {services_with_errors.most_common(1)[0] if services_with_errors else 'None'}"
        ]
        
        # Recommendations
        recommendations = []
        if error_rate > 5:
            recommendations.append("High error rate detected - investigate immediately")
        if 'timeout' in [et.lower() for et in error_types.keys()]:
            recommendations.append("Timeout errors detected - check service performance")
        if 'connection' in [et.lower() for et in error_types.keys()]:
            recommendations.append("Connection errors detected - check network connectivity")
        
        return AnalysisResult(
            analysis_type="error_analysis",
            timestamp=datetime.now(),
            summary={
                "total_errors": total_errors,
                "error_rate": error_rate,
                "error_types": dict(error_types),
                "services_with_errors": dict(services_with_errors),
                "error_timeline": {str(k): v for k, v in error_timeline.items()}
            },
            insights=insights,
            recommendations=recommendations,
            visualizations=["error_timeline.png", "error_types.png"]
        )
    
    def _analyze_performance(self, logs: List[LogEntry]) -> Optional[AnalysisResult]:
        """Analyze performance metrics and latency patterns"""
        
        latency_logs = [log for log in logs if log.latency is not None]
        
        if not latency_logs:
            return None
        
        latencies = [log.latency for log in latency_logs]
        
        # Performance statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        # Service-wise performance
        service_latencies = defaultdict(list)
        for log in latency_logs:
            service_latencies[log.service].append(log.latency)
        
        service_performance = {}
        for service, latencies in service_latencies.items():
            service_performance[service] = {
                "avg": np.mean(latencies),
                "p95": np.percentile(latencies, 95),
                "count": len(latencies)
            }
        
        # High latency detection
        high_latency_threshold = 1000  # 1 second
        high_latency_logs = [log for log in latency_logs if log.latency > high_latency_threshold]
        
        # Insights generation
        insights = [
            f"Average latency: {avg_latency:.2f}ms",
            f"95th percentile latency: {p95_latency:.2f}ms",
            f"99th percentile latency: {p99_latency:.2f}ms",
            f"High latency requests (>{high_latency_threshold}ms): {len(high_latency_logs)}"
        ]
        
        # Recommendations
        recommendations = []
        if p95_latency > 500:
            recommendations.append("High P95 latency - optimize slow endpoints")
        if len(high_latency_logs) > len(latency_logs) * 0.01:
            recommendations.append("More than 1% of requests have high latency")
        
        return AnalysisResult(
            analysis_type="performance_analysis",
            timestamp=datetime.now(),
            summary={
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
                "max_latency": max_latency,
                "high_latency_count": len(high_latency_logs),
                "service_performance": service_performance
            },
            insights=insights,
            recommendations=recommendations,
            visualizations=["latency_distribution.png", "service_performance.png"]
        )
    
    def _analyze_predictions(self, logs: List[LogEntry]) -> Optional[AnalysisResult]:
        """Analyze prediction quality and patterns"""
        
        prediction_logs = [log for log in logs if log.drug_id and log.confidence is not None]
        
        if not prediction_logs:
            return None
        
        confidences = [log.confidence for log in prediction_logs]
        
        # Prediction statistics
        total_predictions = len(prediction_logs)
        avg_confidence = np.mean(confidences)
        low_confidence_threshold = 0.5
        low_confidence_count = len([c for c in confidences if c < low_confidence_threshold])
        
        # Drug and target analysis
        drug_counts = Counter([log.drug_id for log in prediction_logs])
        target_counts = Counter([log.target_id for log in prediction_logs if log.target_id])
        
        # Confidence distribution
        confidence_bins = {
            "very_low": len([c for c in confidences if c < 0.3]),
            "low": len([c for c in confidences if 0.3 <= c < 0.5]),
            "medium": len([c for c in confidences if 0.5 <= c < 0.7]),
            "high": len([c for c in confidences if 0.7 <= c < 0.9]),
            "very_high": len([c for c in confidences if c >= 0.9])
        }
        
        # Insights generation
        insights = [
            f"Total predictions: {total_predictions}",
            f"Average confidence: {avg_confidence:.3f}",
            f"Low confidence predictions (<{low_confidence_threshold}): {low_confidence_count}",
            f"Most predicted drug: {drug_counts.most_common(1)[0] if drug_counts else 'None'}",
            f"Most predicted target: {target_counts.most_common(1)[0] if target_counts else 'None'}"
        ]
        
        # Recommendations
        recommendations = []
        if avg_confidence < 0.7:
            recommendations.append("Average confidence is low - review model performance")
        if low_confidence_count > total_predictions * 0.2:
            recommendations.append("More than 20% of predictions have low confidence")
        
        return AnalysisResult(
            analysis_type="prediction_analysis",
            timestamp=datetime.now(),
            summary={
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "low_confidence_count": low_confidence_count,
                "confidence_distribution": confidence_bins,
                "top_drugs": dict(drug_counts.most_common(10)),
                "top_targets": dict(target_counts.most_common(10))
            },
            insights=insights,
            recommendations=recommendations,
            visualizations=["confidence_distribution.png", "top_predictions.png"]
        )
    
    def _analyze_traffic_patterns(self, logs: List[LogEntry]) -> Optional[AnalysisResult]:
        """Analyze traffic patterns and usage trends"""
        
        # Hourly traffic analysis
        hourly_traffic = defaultdict(int)
        for log in logs:
            hour = log.timestamp.hour
            hourly_traffic[hour] += 1
        
        # Service usage
        service_usage = Counter([log.service for log in logs])
        
        # Peak hours detection
        peak_hour = max(hourly_traffic.items(), key=lambda x: x[1]) if hourly_traffic else (0, 0)
        
        insights = [
            f"Total log entries: {len(logs)}",
            f"Peak hour: {peak_hour[0]}:00 with {peak_hour[1]} entries",
            f"Most active service: {service_usage.most_common(1)[0] if service_usage else 'None'}"
        ]
        
        recommendations = []
        if peak_hour[1] > len(logs) * 0.3:
            recommendations.append("Significant traffic spike detected - consider auto-scaling")
        
        return AnalysisResult(
            analysis_type="traffic_analysis",
            timestamp=datetime.now(),
            summary={
                "total_entries": len(logs),
                "hourly_traffic": dict(hourly_traffic),
                "service_usage": dict(service_usage),
                "peak_hour": peak_hour
            },
            insights=insights,
            recommendations=recommendations,
            visualizations=["hourly_traffic.png", "service_usage.png"]
        )
    
    def _analyze_drift_patterns(self, logs: List[LogEntry]) -> Optional[AnalysisResult]:
        """Analyze drift detection patterns"""
        
        # Filter for drift-related logs
        drift_logs = [log for log in logs if 'drift' in log.message.lower()]
        
        if not drift_logs:
            return None
        
        # Extract drift information (would need more sophisticated parsing)
        drift_detections = len([log for log in drift_logs if 'detected' in log.message.lower()])
        
        insights = [
            f"Drift-related log entries: {len(drift_logs)}",
            f"Drift detections: {drift_detections}"
        ]
        
        recommendations = []
        if drift_detections > 0:
            recommendations.append("Data drift detected - review model performance")
        
        return AnalysisResult(
            analysis_type="drift_analysis",
            timestamp=datetime.now(),
            summary={
                "drift_logs": len(drift_logs),
                "drift_detections": drift_detections
            },
            insights=insights,
            recommendations=recommendations,
            visualizations=[]
        )
    
    def generate_visualizations(self, analysis_results: List[AnalysisResult], 
                              output_dir: str = "analysis_output"):
        """Generate visualizations for analysis results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for result in analysis_results:
            if result.analysis_type == "error_analysis":
                self._plot_error_analysis(result, output_path)
            elif result.analysis_type == "performance_analysis":
                self._plot_performance_analysis(result, output_path)
            elif result.analysis_type == "prediction_analysis":
                self._plot_prediction_analysis(result, output_path)
            elif result.analysis_type == "traffic_analysis":
                self._plot_traffic_analysis(result, output_path)
    
    def _plot_error_analysis(self, result: AnalysisResult, output_path: Path):
        """Generate error analysis plots"""
        summary = result.summary
        
        # Error types pie chart
        if summary["error_types"]:
            plt.figure(figsize=(10, 6))
            plt.pie(summary["error_types"].values(), 
                   labels=summary["error_types"].keys(), 
                   autopct='%1.1f%%')
            plt.title("Error Types Distribution")
            plt.savefig(output_path / "error_types.png")
            plt.close()
        
        # Error timeline
        if summary["error_timeline"]:
            plt.figure(figsize=(12, 6))
            times = list(summary["error_timeline"].keys())
            counts = list(summary["error_timeline"].values())
            plt.plot(times, counts, marker='o')
            plt.title("Error Timeline")
            plt.xlabel("Time")
            plt.ylabel("Error Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "error_timeline.png")
            plt.close()
    
    def _plot_performance_analysis(self, result: AnalysisResult, output_path: Path):
        """Generate performance analysis plots"""
        summary = result.summary
        
        # Service performance comparison
        if summary["service_performance"]:
            services = list(summary["service_performance"].keys())
            avg_latencies = [summary["service_performance"][s]["avg"] for s in services]
            p95_latencies = [summary["service_performance"][s]["p95"] for s in services]
            
            x = np.arange(len(services))
            width = 0.35
            
            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, avg_latencies, width, label='Average')
            plt.bar(x + width/2, p95_latencies, width, label='P95')
            plt.xlabel('Services')
            plt.ylabel('Latency (ms)')
            plt.title('Service Performance Comparison')
            plt.xticks(x, services, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / "service_performance.png")
            plt.close()
    
    def _plot_prediction_analysis(self, result: AnalysisResult, output_path: Path):
        """Generate prediction analysis plots"""
        summary = result.summary
        
        # Confidence distribution
        if summary["confidence_distribution"]:
            plt.figure(figsize=(10, 6))
            bins = list(summary["confidence_distribution"].keys())
            counts = list(summary["confidence_distribution"].values())
            plt.bar(bins, counts)
            plt.title("Prediction Confidence Distribution")
            plt.xlabel("Confidence Range")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "confidence_distribution.png")
            plt.close()
    
    def _plot_traffic_analysis(self, result: AnalysisResult, output_path: Path):
        """Generate traffic analysis plots"""
        summary = result.summary
        
        # Hourly traffic
        if summary["hourly_traffic"]:
            plt.figure(figsize=(12, 6))
            hours = list(summary["hourly_traffic"].keys())
            counts = list(summary["hourly_traffic"].values())
            plt.bar(hours, counts)
            plt.title("Hourly Traffic Pattern")
            plt.xlabel("Hour of Day")
            plt.ylabel("Request Count")
            plt.savefig(output_path / "hourly_traffic.png")
            plt.close()
    
    def export_analysis_report(self, analysis_results: List[AnalysisResult], 
                              output_file: str = "analysis_report.json"):
        """Export analysis results to JSON report"""
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_analyses": len(analysis_results),
            "results": [result.to_dict() for result in analysis_results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report exported to {output_file}")

if __name__ == "__main__":
    # Example usage
    analyzer = LogAnalyzer()
    
    # Analyze logs from the last hour
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    results = analyzer.analyze_logs(start_time, end_time)
    
    if results:
        print(f"Generated {len(results)} analysis results:")
        for result in results:
            print(f"- {result.analysis_type}: {len(result.insights)} insights")
        
        # Generate visualizations
        analyzer.generate_visualizations(results)
        
        # Export report
        analyzer.export_analysis_report(results)
    else:
        print("No analysis results generated")