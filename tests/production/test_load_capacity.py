#!/usr/bin/env python3
"""
Production Load Capacity Testing
Stress testing and capacity validation for DrugBAN API in production.
"""

import argparse
import json
import sys
import time
import requests
import statistics
import threading
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, deque
import psutil
import matplotlib.pyplot as plt

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LoadCapacityTester:
    """Comprehensive load capacity testing for production API."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize load capacity tester.
        
        Args:
            base_url: Base URL of the production API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._setup_session()
        self.auth_token: Optional[str] = None
        
        # Results storage
        self.results: Dict[str, Any] = {
            "test_timestamp": time.time(),
            "test_environment": "production_load",
            "test_configuration": {
                "base_url": base_url,
                "timeout": timeout
            },
            "load_tests": {},
            "capacity_metrics": {},
            "recommendations": []
        }
        
        # Real-time metrics
        self.metrics_lock = threading.Lock()
        self.real_time_metrics = {
            "requests_sent": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "response_times": deque(maxlen=1000),
            "status_codes": defaultdict(int),
            "current_rps": 0,
            "peak_rps": 0
        }
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session optimized for load testing."""
        session = requests.Session()
        
        # Aggressive retry strategy for load testing
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=0.5
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def authenticate(self) -> bool:
        """Authenticate with the production API."""
        try:
            auth_data = {
                "username": "load_test_user",
                "password": "load_test_password"
            }
            
            response = self.session.post(
                f"{self.base_url}/auth/token",
                json=auth_data,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                if "access_token" in result:
                    self.auth_token = result["access_token"]
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _make_prediction_request(self, request_id: int) -> Dict[str, Any]:
        """Make a single prediction request and return metrics."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        test_request = {
            "drug_info": {
                "smiles": "CCO",  # Simple test molecule
                "drug_id": f"LOAD_TEST_DRUG_{request_id}"
            },
            "target_info": {
                "target_id": f"LOAD_TEST_TARGET_{request_id}",
                "target_class": "enzyme"
            }
        }
        
        try:
            start_time = time.time()
            
            with self.metrics_lock:
                self.real_time_metrics["requests_sent"] += 1
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_request,
                headers=headers,
                timeout=self.timeout,
                verify=False
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update metrics
            with self.metrics_lock:
                self.real_time_metrics["requests_completed"] += 1
                self.real_time_metrics["response_times"].append(response_time)
                self.real_time_metrics["status_codes"][response.status_code] += 1
                
                if response.status_code != 200:
                    self.real_time_metrics["requests_failed"] += 1
            
            return {
                "request_id": request_id,
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response_time,
                "timestamp": start_time,
                "response_size": len(response.content) if response.content else 0
            }
            
        except Exception as e:
            with self.metrics_lock:
                self.real_time_metrics["requests_completed"] += 1
                self.real_time_metrics["requests_failed"] += 1
                self.real_time_metrics["status_codes"][0] += 1
            
            return {
                "request_id": request_id,
                "success": False,
                "status_code": 0,
                "response_time": self.timeout,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    def _calculate_rps(self, start_time: float, window_size: int = 10):
        """Calculate requests per second in a background thread."""
        while True:
            time.sleep(1)
            
            current_time = time.time()
            with self.metrics_lock:
                # Count recent requests
                recent_requests = sum(1 for rt in self.real_time_metrics["response_times"] 
                                    if current_time - rt <= window_size)
                current_rps = recent_requests / window_size
                
                self.real_time_metrics["current_rps"] = current_rps
                if current_rps > self.real_time_metrics["peak_rps"]:
                    self.real_time_metrics["peak_rps"] = current_rps
            
            # Stop if test is complete (no requests in last 30 seconds)
            if current_time - start_time > 30 and current_rps == 0:
                break
    
    def test_sustained_load(self, target_rps: int, duration_seconds: int = 300) -> Dict[str, Any]:
        """Test sustained load at target RPS for specified duration."""
        print(f"🔄 Testing sustained load: {target_rps} RPS for {duration_seconds} seconds")
        
        # Reset metrics
        with self.metrics_lock:
            self.real_time_metrics = {
                "requests_sent": 0,
                "requests_completed": 0,
                "requests_failed": 0,
                "response_times": deque(maxlen=1000),
                "status_codes": defaultdict(int),
                "current_rps": 0,
                "peak_rps": 0
            }
        
        start_time = time.time()
        results = []
        request_id = 0
        
        # Start RPS calculation thread
        rps_thread = threading.Thread(target=self._calculate_rps, args=(start_time,))
        rps_thread.daemon = True
        rps_thread.start()
        
        # Calculate inter-request delay
        delay_between_requests = 1.0 / target_rps if target_rps > 0 else 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(target_rps * 2, 100)) as executor:
            end_time = start_time + duration_seconds
            
            while time.time() < end_time:
                request_start = time.time()
                
                # Submit request
                future = executor.submit(self._make_prediction_request, request_id)
                results.append(future)
                request_id += 1
                
                # Control request rate
                elapsed = time.time() - request_start
                if elapsed < delay_between_requests:
                    time.sleep(delay_between_requests - elapsed)
                
                # Print progress every 30 seconds
                if request_id % (target_rps * 30) == 0:
                    with self.metrics_lock:
                        current_rps = self.real_time_metrics["current_rps"]
                        success_rate = (self.real_time_metrics["requests_completed"] - 
                                      self.real_time_metrics["requests_failed"]) / max(1, self.real_time_metrics["requests_completed"])
                    
                    print(f"   Progress: {request_id} requests sent, "
                          f"Current RPS: {current_rps:.1f}, "
                          f"Success rate: {success_rate:.1%}")
            
            # Wait for all requests to complete
            print("   Waiting for requests to complete...")
            completed_results = []
            for future in concurrent.futures.as_completed(results, timeout=self.timeout * 2):
                try:
                    completed_results.append(future.result())
                except Exception as e:
                    completed_results.append({
                        "success": False,
                        "error": str(e),
                        "response_time": self.timeout
                    })
        
        # Analyze results
        total_requests = len(completed_results)
        successful_requests = sum(1 for r in completed_results if r.get("success", False))
        response_times = [r["response_time"] for r in completed_results if "response_time" in r]
        
        test_duration = time.time() - start_time
        actual_rps = total_requests / test_duration
        
        return {
            "test_type": "sustained_load",
            "target_rps": target_rps,
            "duration_seconds": duration_seconds,
            "actual_duration": test_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "actual_rps": actual_rps,
            "rps_accuracy": actual_rps / target_rps if target_rps > 0 else 0,
            "response_time_stats": {
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p95": sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0,
                "p99": sorted(response_times)[int(0.99 * len(response_times))] if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0
            },
            "status_code_distribution": dict(self.real_time_metrics["status_codes"]),
            "peak_rps_achieved": self.real_time_metrics["peak_rps"]
        }
    
    def test_capacity_limits(self, max_rps: int = 100, step_size: int = 10, 
                           step_duration: int = 60) -> Dict[str, Any]:
        """Test capacity limits by gradually increasing load."""
        print(f"📈 Testing capacity limits: 0 to {max_rps} RPS in steps of {step_size}")
        
        capacity_results = []
        breaking_point = None
        
        for target_rps in range(step_size, max_rps + 1, step_size):
            print(f"\n   Testing {target_rps} RPS...")
            
            result = self.test_sustained_load(target_rps, step_duration)
            capacity_results.append(result)
            
            # Check if we've reached the breaking point
            success_rate = result["success_rate"]
            avg_response_time = result["response_time_stats"]["mean"]
            
            if success_rate < 0.95 or avg_response_time > 5.0:
                breaking_point = target_rps
                print(f"   ⚠️  Breaking point detected at {target_rps} RPS")
                break
            
            # Brief cooldown between tests
            print(f"   ✅ {target_rps} RPS: {success_rate:.1%} success, {avg_response_time:.2f}s avg response")
            time.sleep(5)
        
        return {
            "test_type": "capacity_limits",
            "max_rps_tested": max_rps,
            "step_size": step_size,
            "step_duration": step_duration,
            "capacity_results": capacity_results,
            "breaking_point_rps": breaking_point,
            "recommended_max_rps": int(breaking_point * 0.8) if breaking_point else max_rps
        }
    
    def test_spike_handling(self, baseline_rps: int = 10, spike_rps: int = 50, 
                           spike_duration: int = 30) -> Dict[str, Any]:
        """Test how the API handles traffic spikes."""
        print(f"⚡ Testing spike handling: {baseline_rps} RPS → {spike_rps} RPS → {baseline_rps} RPS")
        
        phases = [
            ("baseline_before", baseline_rps, 60),
            ("spike", spike_rps, spike_duration),
            ("baseline_after", baseline_rps, 60)
        ]
        
        spike_results = {}
        
        for phase_name, rps, duration in phases:
            print(f"   Phase: {phase_name} ({rps} RPS for {duration}s)")
            result = self.test_sustained_load(rps, duration)
            spike_results[phase_name] = result
            
            # Brief pause between phases
            time.sleep(2)
        
        # Analyze spike recovery
        baseline_before_success = spike_results["baseline_before"]["success_rate"]
        spike_success = spike_results["spike"]["success_rate"]
        baseline_after_success = spike_results["baseline_after"]["success_rate"]
        
        recovery_time = 0  # Would need more detailed analysis
        
        return {
            "test_type": "spike_handling",
            "baseline_rps": baseline_rps,
            "spike_rps": spike_rps,
            "spike_duration": spike_duration,
            "phases": spike_results,
            "spike_impact": {
                "baseline_before_success_rate": baseline_before_success,
                "spike_success_rate": spike_success,
                "baseline_after_success_rate": baseline_after_success,
                "performance_degradation": max(0, baseline_before_success - spike_success),
                "recovery_complete": abs(baseline_after_success - baseline_before_success) < 0.05,
                "recovery_time_estimate": recovery_time
            }
        }
    
    def test_concurrent_users(self, user_counts: List[int] = [1, 5, 10, 25, 50]) -> Dict[str, Any]:
        """Test API performance with different numbers of concurrent users."""
        print(f"👥 Testing concurrent users: {user_counts}")
        
        concurrent_results = {}
        
        for user_count in user_counts:
            print(f"\n   Testing {user_count} concurrent users...")
            
            # Each user makes 10 requests
            requests_per_user = 10
            total_requests = user_count * requests_per_user
            
            results = []
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=user_count) as executor:
                # Submit all requests
                futures = []
                for user_id in range(user_count):
                    for req_id in range(requests_per_user):
                        request_id = user_id * requests_per_user + req_id
                        future = executor.submit(self._make_prediction_request, request_id)
                        futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "response_time": self.timeout
                        })
            
            end_time = time.time()
            test_duration = end_time - start_time
            
            # Analyze results
            successful_requests = sum(1 for r in results if r.get("success", False))
            response_times = [r["response_time"] for r in results if "response_time" in r]
            
            concurrent_results[str(user_count)] = {
                "user_count": user_count,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "test_duration": test_duration,
                "requests_per_second": total_requests / test_duration,
                "average_response_time": statistics.mean(response_times) if response_times else 0,
                "median_response_time": statistics.median(response_times) if response_times else 0,
                "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0
            }
            
            print(f"   ✅ {user_count} users: {concurrent_results[str(user_count)]['success_rate']:.1%} success")
        
        return {
            "test_type": "concurrent_users",
            "user_counts_tested": user_counts,
            "results": concurrent_results
        }
    
    def run_comprehensive_load_tests(self) -> Dict[str, Any]:
        """Run comprehensive load capacity tests."""
        print("🚀 Starting comprehensive load capacity tests...\n")
        
        # Test 1: Sustained Load Test
        print("Test 1: Sustained Load Performance")
        sustained_result = self.test_sustained_load(target_rps=20, duration_seconds=120)
        self.results["load_tests"]["sustained_load"] = sustained_result
        
        # Test 2: Capacity Limits
        print("\nTest 2: Capacity Limits Discovery")
        capacity_result = self.test_capacity_limits(max_rps=50, step_size=5, step_duration=30)
        self.results["load_tests"]["capacity_limits"] = capacity_result
        
        # Test 3: Spike Handling
        print("\nTest 3: Traffic Spike Handling")
        spike_result = self.test_spike_handling(baseline_rps=5, spike_rps=25, spike_duration=20)
        self.results["load_tests"]["spike_handling"] = spike_result
        
        # Test 4: Concurrent Users
        print("\nTest 4: Concurrent User Performance")
        concurrent_result = self.test_concurrent_users([1, 5, 10, 20])
        self.results["load_tests"]["concurrent_users"] = concurrent_result
        
        # Generate capacity metrics and recommendations
        self._generate_capacity_metrics()
        self._generate_recommendations()
        
        print(f"\n📊 Load testing completed. Capacity metrics generated.")
        
        return self.results
    
    def _generate_capacity_metrics(self):
        """Generate overall capacity metrics from test results."""
        capacity_limits = self.results["load_tests"].get("capacity_limits", {})
        sustained_load = self.results["load_tests"].get("sustained_load", {})
        spike_handling = self.results["load_tests"].get("spike_handling", {})
        
        self.results["capacity_metrics"] = {
            "max_sustainable_rps": capacity_limits.get("recommended_max_rps", 0),
            "breaking_point_rps": capacity_limits.get("breaking_point_rps", 0),
            "sustained_performance": {
                "rps": sustained_load.get("actual_rps", 0),
                "success_rate": sustained_load.get("success_rate", 0),
                "avg_response_time": sustained_load.get("response_time_stats", {}).get("mean", 0)
            },
            "spike_resilience": {
                "can_handle_spikes": spike_handling.get("spike_impact", {}).get("recovery_complete", False),
                "performance_degradation": spike_handling.get("spike_impact", {}).get("performance_degradation", 1.0)
            }
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        capacity_metrics = self.results["capacity_metrics"]
        
        # RPS recommendations
        max_rps = capacity_metrics.get("max_sustainable_rps", 0)
        if max_rps > 0:
            recommendations.append(f"Recommended maximum RPS for production: {max_rps}")
            recommendations.append(f"Set autoscaling threshold at {int(max_rps * 0.7)} RPS")
        
        # Performance recommendations
        sustained_perf = capacity_metrics.get("sustained_performance", {})
        if sustained_perf.get("avg_response_time", 0) > 2.0:
            recommendations.append("Average response time exceeds 2s - consider performance optimization")
        
        if sustained_perf.get("success_rate", 0) < 0.99:
            recommendations.append("Success rate below 99% - investigate error sources")
        
        # Spike handling recommendations
        spike_resilience = capacity_metrics.get("spike_resilience", {})
        if not spike_resilience.get("can_handle_spikes", False):
            recommendations.append("API struggles with traffic spikes - implement better autoscaling")
        
        degradation = spike_resilience.get("performance_degradation", 0)
        if degradation > 0.1:
            recommendations.append(f"Significant performance degradation during spikes ({degradation:.1%}) - consider circuit breakers")
        
        self.results["recommendations"] = recommendations


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test DrugBAN API load capacity in production")
    parser.add_argument("--base-url", required=True, help="Base URL of the production API")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    parser.add_argument("--max-rps", type=int, default=50, help="Maximum RPS to test")
    parser.add_argument("--quick", action="store_true", help="Run quick tests with reduced duration")
    
    args = parser.parse_args()
    
    # Run tests
    tester = LoadCapacityTester(args.base_url, args.timeout)
    
    # Try to authenticate (optional)
    tester.authenticate()
    
    # Adjust test parameters for quick mode
    if args.quick:
        print("🏃 Running in quick mode...")
        # Quick sustained load test
        sustained_result = tester.test_sustained_load(target_rps=10, duration_seconds=30)
        tester.results["load_tests"]["sustained_load"] = sustained_result
        
        # Quick capacity test
        capacity_result = tester.test_capacity_limits(max_rps=20, step_size=5, step_duration=15)
        tester.results["load_tests"]["capacity_limits"] = capacity_result
        
        tester._generate_capacity_metrics()
        tester._generate_recommendations()
    else:
        results = tester.run_comprehensive_load_tests()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(tester.results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Print summary
    capacity_metrics = tester.results.get("capacity_metrics", {})
    max_rps = capacity_metrics.get("max_sustainable_rps", 0)
    
    print(f"\n📈 Load Capacity Summary:")
    print(f"   Maximum sustainable RPS: {max_rps}")
    print(f"   Breaking point: {capacity_metrics.get('breaking_point_rps', 'Not reached')}")
    
    # Print recommendations
    recommendations = tester.results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Success criteria: API should handle at least 10 RPS
    if max_rps >= 10:
        print(f"\n✅ Load capacity tests PASSED (sustainable RPS: {max_rps})")
        sys.exit(0)
    else:
        print(f"\n❌ Load capacity tests FAILED (sustainable RPS: {max_rps}, minimum required: 10)")
        sys.exit(1)


if __name__ == "__main__":
    main()