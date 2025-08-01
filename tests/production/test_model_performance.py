#!/usr/bin/env python3
"""
Production Model Performance Testing
Comprehensive performance validation for DrugBAN model in production.
"""

import argparse
import json
import sys
import time
import requests
import statistics
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ProductionModelTester:
    """Comprehensive model performance testing for production."""
    
    def __init__(self, base_url: str, timeout: int = 60):
        """Initialize production model tester.
        
        Args:
            base_url: Base URL of the production API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._setup_session()
        self.auth_token: Optional[str] = None
        self.results: Dict[str, Any] = {
            "test_timestamp": time.time(),
            "test_environment": "production",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "performance_metrics": {},
            "test_results": []
        }
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with production-grade retry strategy."""
        session = requests.Session()
        
        # More aggressive retry strategy for production
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=2
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def authenticate(self) -> bool:
        """Authenticate with the production API."""
        try:
            # Try to get token (this would use real credentials in production)
            auth_data = {
                "username": "production_test_user",
                "password": "production_test_password"
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
    
    def test_model_availability(self) -> Dict[str, Any]:
        """Test model availability and basic info."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response = self.session.get(
                f"{self.base_url}/model/info",
                headers=headers,
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    "available": True,
                    "model_info": model_info,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "available": False,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
                
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "response_time": self.timeout
            }
    
    def test_prediction_accuracy(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Test prediction accuracy with known test cases."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        response_times = []
        errors = []
        
        for sample in test_samples:
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=sample["input"],
                    headers=headers,
                    timeout=self.timeout,
                    verify=False
                )
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    confidence = result.get("confidence", 0)
                    
                    confidence_scores.append(confidence)
                    total_predictions += 1
                    
                    # Check if prediction matches expected (if available)
                    if "expected" in sample and prediction == sample["expected"]:
                        correct_predictions += 1
                else:
                    errors.append(f"HTTP {response.status_code}")
                    
            except Exception as e:
                errors.append(str(e))
                response_times.append(self.timeout)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return {
            "accuracy": accuracy,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "average_confidence": avg_confidence,
            "average_response_time": avg_response_time,
            "max_response_time": max(response_times) if response_times else 0,
            "error_count": len(errors),
            "errors": errors[:5]  # First 5 errors
        }
    
    def test_load_performance(self, concurrent_requests: int = 10, 
                             total_requests: int = 100) -> Dict[str, Any]:
        """Test API performance under load."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Sample prediction request
        test_request = {
            "drug_info": {
                "smiles": "CCO",
                "drug_id": "LOAD_TEST_DRUG"
            },
            "target_info": {
                "target_id": "LOAD_TEST_TARGET",
                "target_class": "enzyme"
            }
        }
        
        def make_request() -> Tuple[bool, float, int]:
            """Make a single request and return (success, response_time, status_code)."""
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_request,
                    headers=headers,
                    timeout=self.timeout,
                    verify=False
                )
                end_time = time.time()
                
                return (
                    response.status_code == 200,
                    end_time - start_time,
                    response.status_code
                )
            except Exception:
                return False, self.timeout, 0
        
        # Execute concurrent requests
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        
        # Analyze results
        successful_requests = sum(1 for success, _, _ in results if success)
        response_times = [rt for _, rt, _ in results]
        status_codes = [sc for _, _, sc in results]
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests,
            "total_duration": end_time - start_time,
            "requests_per_second": total_requests / (end_time - start_time),
            "average_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
            "p99_response_time": sorted(response_times)[int(0.99 * len(response_times))],
            "max_response_time": max(response_times),
            "min_response_time": min(response_times),
            "status_code_distribution": {
                str(code): status_codes.count(code) for code in set(status_codes)
            }
        }
    
    def test_batch_performance(self, batch_sizes: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Test batch prediction performance."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            # Create batch request
            predictions = []
            for i in range(batch_size):
                predictions.append({
                    "drug_info": {
                        "smiles": "CCO",
                        "drug_id": f"BATCH_TEST_DRUG_{i}"
                    },
                    "target_info": {
                        "target_id": f"BATCH_TEST_TARGET_{i}",
                        "target_class": "enzyme"
                    }
                })
            
            batch_request = {"predictions": predictions}
            
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict/batch",
                    json=batch_request,
                    headers=headers,
                    timeout=self.timeout * 2,  # Longer timeout for batch
                    verify=False
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    predictions_returned = len(result.get("predictions", []))
                    
                    batch_results[str(batch_size)] = {
                        "success": True,
                        "response_time": end_time - start_time,
                        "predictions_requested": batch_size,
                        "predictions_returned": predictions_returned,
                        "predictions_per_second": batch_size / (end_time - start_time),
                        "avg_time_per_prediction": (end_time - start_time) / batch_size
                    }
                else:
                    batch_results[str(batch_size)] = {
                        "success": False,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time
                    }
                    
            except Exception as e:
                batch_results[str(batch_size)] = {
                    "success": False,
                    "error": str(e),
                    "response_time": self.timeout * 2
                }
        
        return batch_results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test API error handling with invalid inputs."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        error_tests = [
            {
                "name": "invalid_smiles",
                "request": {
                    "drug_info": {
                        "smiles": "INVALID_SMILES_STRING",
                        "drug_id": "ERROR_TEST_001"
                    },
                    "target_info": {
                        "target_id": "ERROR_TEST_TARGET",
                        "target_class": "enzyme"
                    }
                },
                "expected_status": [400, 422]
            },
            {
                "name": "missing_required_fields",
                "request": {
                    "drug_info": {
                        "drug_id": "ERROR_TEST_002"
                        # Missing smiles
                    },
                    "target_info": {
                        "target_id": "ERROR_TEST_TARGET"
                        # Missing target_class
                    }
                },
                "expected_status": [400, 422]
            },
            {
                "name": "empty_request",
                "request": {},
                "expected_status": [400, 422]
            }
        ]
        
        error_results = {}
        
        for test in error_tests:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test["request"],
                    headers=headers,
                    timeout=self.timeout,
                    verify=False
                )
                
                error_results[test["name"]] = {
                    "status_code": response.status_code,
                    "correct_error_handling": response.status_code in test["expected_status"],
                    "response_time": response.elapsed.total_seconds(),
                    "has_error_message": len(response.text) > 0
                }
                
            except Exception as e:
                error_results[test["name"]] = {
                    "error": str(e),
                    "correct_error_handling": False
                }
        
        return error_results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all production model performance tests."""
        print("🏭 Starting production model performance tests...\n")
        
        # Test 1: Model Availability
        print("1. Testing model availability...")
        availability_result = self.test_model_availability()
        self.results["model_availability"] = availability_result
        
        if not availability_result["available"]:
            print("❌ Model not available - skipping remaining tests")
            return self.results
        
        print(f"✅ Model available (response time: {availability_result['response_time']:.3f}s)")
        
        # Test 2: Prediction Accuracy (with sample data)
        print("\n2. Testing prediction accuracy...")
        test_samples = [
            {
                "input": {
                    "drug_info": {"smiles": "CCO", "drug_id": "TEST_001"},
                    "target_info": {"target_id": "TEST_TARGET_001", "target_class": "enzyme"}
                }
            },
            {
                "input": {
                    "drug_info": {"smiles": "CC(C)O", "drug_id": "TEST_002"},
                    "target_info": {"target_id": "TEST_TARGET_002", "target_class": "receptor"}
                }
            },
            {
                "input": {
                    "drug_info": {"smiles": "CC(C)(C)O", "drug_id": "TEST_003"},
                    "target_info": {"target_id": "TEST_TARGET_003", "target_class": "channel"}
                }
            }
        ]
        
        accuracy_result = self.test_prediction_accuracy(test_samples)
        self.results["prediction_accuracy"] = accuracy_result
        print(f"✅ Accuracy test completed (avg confidence: {accuracy_result['average_confidence']:.3f})")
        
        # Test 3: Load Performance
        print("\n3. Testing load performance...")
        load_result = self.test_load_performance(concurrent_requests=5, total_requests=50)
        self.results["load_performance"] = load_result
        print(f"✅ Load test completed (success rate: {load_result['success_rate']:.1%}, RPS: {load_result['requests_per_second']:.1f})")
        
        # Test 4: Batch Performance
        print("\n4. Testing batch performance...")
        batch_result = self.test_batch_performance([1, 5, 10])
        self.results["batch_performance"] = batch_result
        print("✅ Batch test completed")
        
        # Test 5: Error Handling
        print("\n5. Testing error handling...")
        error_result = self.test_error_handling()
        self.results["error_handling"] = error_result
        print("✅ Error handling test completed")
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score()
        self.results["overall_performance_score"] = performance_score
        
        print(f"\n📊 Overall Performance Score: {performance_score:.1f}/100")
        
        return self.results
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score based on test results."""
        score = 0.0
        
        # Model availability (20 points)
        if self.results.get("model_availability", {}).get("available", False):
            score += 20
        
        # Load performance (30 points)
        load_perf = self.results.get("load_performance", {})
        if load_perf.get("success_rate", 0) >= 0.95:
            score += 30
        elif load_perf.get("success_rate", 0) >= 0.90:
            score += 20
        elif load_perf.get("success_rate", 0) >= 0.80:
            score += 10
        
        # Response time (25 points)
        avg_response_time = load_perf.get("average_response_time", 999)
        if avg_response_time <= 1.0:
            score += 25
        elif avg_response_time <= 2.0:
            score += 20
        elif avg_response_time <= 3.0:
            score += 15
        elif avg_response_time <= 5.0:
            score += 10
        
        # Error handling (15 points)
        error_handling = self.results.get("error_handling", {})
        correct_errors = sum(1 for result in error_handling.values() 
                           if isinstance(result, dict) and result.get("correct_error_handling", False))
        total_error_tests = len(error_handling)
        if total_error_tests > 0:
            score += 15 * (correct_errors / total_error_tests)
        
        # Prediction quality (10 points)
        pred_accuracy = self.results.get("prediction_accuracy", {})
        if pred_accuracy.get("error_count", 999) == 0:
            score += 10
        elif pred_accuracy.get("error_count", 999) <= 1:
            score += 5
        
        return min(score, 100.0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test DrugBAN model performance in production")
    parser.add_argument("--base-url", required=True, help="Base URL of the production API")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    parser.add_argument("--min-score", type=float, default=80.0, 
                       help="Minimum performance score required (default: 80.0)")
    
    args = parser.parse_args()
    
    # Run tests
    tester = ProductionModelTester(args.base_url, args.timeout)
    
    # Try to authenticate (optional for this test)
    tester.authenticate()
    
    results = tester.run_comprehensive_tests()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Check if score meets requirement
    performance_score = results.get("overall_performance_score", 0)
    if performance_score >= args.min_score:
        print(f"\n✅ Production performance tests PASSED (score: {performance_score:.1f})")
        sys.exit(0)
    else:
        print(f"\n❌ Production performance tests FAILED (score: {performance_score:.1f}, required: {args.min_score})")
        sys.exit(1)


if __name__ == "__main__":
    main()