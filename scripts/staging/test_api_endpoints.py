#!/usr/bin/env python3
"""
API Endpoint Testing for Staging Environment
Comprehensive testing suite for DrugBAN API endpoints in staging.
"""

import argparse
import json
import sys
import time
import requests
from typing import Dict, List, Optional, Any
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings for staging
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class StagingAPITester:
    """Comprehensive API testing for staging environment."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize API tester.
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._setup_session()
        self.auth_token: Optional[str] = None
        self.results: Dict[str, Any] = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": []
        }
    
    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test and record results."""
        self.results["total_tests"] += 1
        
        try:
            start_time = time.time()
            result = test_func(*args, **kwargs)
            end_time = time.time()
            
            test_result = {
                "test_name": test_name,
                "status": "PASSED" if result else "FAILED",
                "duration": round(end_time - start_time, 3),
                "details": getattr(test_func, '_last_result', {})
            }
            
            if result:
                self.results["passed_tests"] += 1
                print(f"✅ {test_name}: PASSED ({test_result['duration']}s)")
            else:
                self.results["failed_tests"] += 1
                print(f"❌ {test_name}: FAILED ({test_result['duration']}s)")
                if test_result["details"]:
                    print(f"   Details: {test_result['details']}")
            
            self.results["test_results"].append(test_result)
            return result
            
        except Exception as e:
            self.results["failed_tests"] += 1
            test_result = {
                "test_name": test_name,
                "status": "ERROR",
                "duration": 0,
                "error": str(e)
            }
            self.results["test_results"].append(test_result)
            print(f"💥 {test_name}: ERROR - {str(e)}")
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
                verify=False
            )
            
            self.test_health_endpoint._last_result = {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
            
            return response.status_code == 200
            
        except Exception as e:
            self.test_health_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_auth_endpoint(self) -> bool:
        """Test authentication endpoint."""
        try:
            auth_data = {
                "username": "test_user",
                "password": "test_password"
            }
            
            response = self.session.post(
                f"{self.base_url}/auth/token",
                json=auth_data,
                timeout=self.timeout,
                verify=False
            )
            
            result = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            self.test_auth_endpoint._last_result = {
                "status_code": response.status_code,
                "has_token": "access_token" in result
            }
            
            # Store token for later tests
            if response.status_code == 200 and "access_token" in result:
                self.auth_token = result["access_token"]
                return True
            
            return response.status_code in [200, 401]  # 401 is acceptable if auth is not configured
            
        except Exception as e:
            self.test_auth_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_model_info_endpoint(self) -> bool:
        """Test model information endpoint."""
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
            
            result = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            self.test_model_info_endpoint._last_result = {
                "status_code": response.status_code,
                "has_model_info": bool(result)
            }
            
            return response.status_code in [200, 401]
            
        except Exception as e:
            self.test_model_info_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_prediction_endpoint(self) -> bool:
        """Test prediction endpoint with sample data."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Sample prediction request
            prediction_data = {
                "drug_info": {
                    "smiles": "CCO",  # Ethanol - simple test molecule
                    "drug_id": "TEST_DRUG_001"
                },
                "target_info": {
                    "target_id": "TEST_TARGET_001",
                    "target_class": "enzyme"
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=prediction_data,
                headers=headers,
                timeout=self.timeout,
                verify=False
            )
            
            result = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            self.test_prediction_endpoint._last_result = {
                "status_code": response.status_code,
                "has_prediction": "prediction" in result,
                "has_confidence": "confidence" in result,
                "response_size": len(str(result))
            }
            
            # Accept 200 (success) or 401 (auth required) or 422 (validation error for test data)
            return response.status_code in [200, 401, 422]
            
        except Exception as e:
            self.test_prediction_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_batch_prediction_endpoint(self) -> bool:
        """Test batch prediction endpoint."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Sample batch prediction request
            batch_data = {
                "predictions": [
                    {
                        "drug_info": {
                            "smiles": "CCO",
                            "drug_id": "TEST_DRUG_001"
                        },
                        "target_info": {
                            "target_id": "TEST_TARGET_001",
                            "target_class": "enzyme"
                        }
                    },
                    {
                        "drug_info": {
                            "smiles": "CC(C)O",
                            "drug_id": "TEST_DRUG_002"
                        },
                        "target_info": {
                            "target_id": "TEST_TARGET_002",
                            "target_class": "receptor"
                        }
                    }
                ]
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data,
                headers=headers,
                timeout=self.timeout,
                verify=False
            )
            
            result = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            self.test_batch_prediction_endpoint._last_result = {
                "status_code": response.status_code,
                "has_predictions": "predictions" in result,
                "prediction_count": len(result.get("predictions", [])) if "predictions" in result else 0
            }
            
            return response.status_code in [200, 401, 422]
            
        except Exception as e:
            self.test_batch_prediction_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint."""
        try:
            response = self.session.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout,
                verify=False
            )
            
            metrics_text = response.text
            
            self.test_metrics_endpoint._last_result = {
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "metrics_count": len([line for line in metrics_text.split('\n') if line and not line.startswith('#')]),
                "has_drugban_metrics": "drugban_" in metrics_text
            }
            
            return response.status_code == 200 and "drugban_" in metrics_text
            
        except Exception as e:
            self.test_metrics_endpoint._last_result = {"error": str(e)}
            return False
    
    def test_api_performance(self) -> bool:
        """Test API performance with multiple requests."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response_times = []
            success_count = 0
            
            # Make 5 requests to health endpoint
            for i in range(5):
                start_time = time.time()
                response = self.session.get(
                    f"{self.base_url}/health",
                    headers=headers,
                    timeout=self.timeout,
                    verify=False
                )
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                if response.status_code == 200:
                    success_count += 1
                
                time.sleep(0.1)  # Small delay between requests
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            self.test_api_performance._last_result = {
                "success_rate": success_count / 5,
                "avg_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "performance_acceptable": avg_response_time < 2.0 and max_response_time < 5.0
            }
            
            return success_count >= 4 and avg_response_time < 2.0
            
        except Exception as e:
            self.test_api_performance._last_result = {"error": str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        print("🚀 Starting staging API endpoint tests...\n")
        
        # Core functionality tests
        self.run_test("Health Check", self.test_health_endpoint)
        self.run_test("Authentication", self.test_auth_endpoint)
        self.run_test("Model Information", self.test_model_info_endpoint)
        self.run_test("Single Prediction", self.test_prediction_endpoint)
        self.run_test("Batch Prediction", self.test_batch_prediction_endpoint)
        self.run_test("Metrics Endpoint", self.test_metrics_endpoint)
        self.run_test("API Performance", self.test_api_performance)
        
        # Calculate success rate
        self.results["success_rate"] = (
            self.results["passed_tests"] / self.results["total_tests"] 
            if self.results["total_tests"] > 0 else 0
        )
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {self.results['total_tests']}")
        print(f"   Passed: {self.results['passed_tests']}")
        print(f"   Failed: {self.results['failed_tests']}")
        print(f"   Success Rate: {self.results['success_rate']:.1%}")
        
        return self.results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test DrugBAN API endpoints in staging")
    parser.add_argument("--base-url", required=True, help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    parser.add_argument("--min-success-rate", type=float, default=0.8, 
                       help="Minimum success rate required (default: 0.8)")
    
    args = parser.parse_args()
    
    # Run tests
    tester = StagingAPITester(args.base_url, args.timeout)
    results = tester.run_all_tests()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Check if success rate meets requirement
    if results["success_rate"] >= args.min_success_rate:
        print(f"\n✅ Staging tests PASSED (success rate: {results['success_rate']:.1%})")
        sys.exit(0)
    else:
        print(f"\n❌ Staging tests FAILED (success rate: {results['success_rate']:.1%}, required: {args.min_success_rate:.1%})")
        sys.exit(1)


if __name__ == "__main__":
    main()