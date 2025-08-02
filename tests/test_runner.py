"""
Test Runner for ML Features

Comprehensive test runner that executes all ML feature tests and generates reports.
"""

import pytest
import sys
import os
import time
from pathlib import Path
from typing import Dict, List
import json


def run_test_suite(test_category: str = "all", verbose: bool = True) -> Dict:
    """
    Run test suite for ML features.
    
    Args:
        test_category: Category of tests to run ("ml_api", "ml_models", "ml_service", 
                      "ml_workflows", "rag_regression", "all")
        verbose: Whether to run in verbose mode
    
    Returns:
        Dictionary with test results
    """
    print(f"\n🚀 Starting ML Features Test Suite - Category: {test_category}")
    print("="*60)
    
    # Define test files mapping
    test_files = {
        "ml_api": ["test_ml_api.py"],
        "ml_models": ["test_ml_models.py"],
        "ml_service": ["test_ml_service.py"],
        "ml_workflows": ["test_ml_workflows.py"],
        "rag_regression": ["test_rag_regression.py"],
        "existing": ["test_auth.py", "test_auth_router.py", "test_evaluation.py"],
        "all": [
            "test_ml_api.py",
            "test_ml_models.py", 
            "test_ml_service.py",
            "test_ml_workflows.py",
            "test_rag_regression.py"
        ]
    }
    
    if test_category not in test_files:
        raise ValueError(f"Invalid test category: {test_category}")
    
    # Get test files to run
    files_to_run = test_files[test_category]
    
    # Build pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Add coverage if running all tests
    if test_category == "all":
        pytest_args.extend([
            "--cov=app",
            "--cov=workflows", 
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Add specific test files
    test_dir = Path(__file__).parent
    for test_file in files_to_run:
        test_path = test_dir / test_file
        if test_path.exists():
            pytest_args.append(str(test_path))
        else:
            print(f"⚠️ Warning: Test file {test_file} not found")
    
    # Add output options
    pytest_args.extend([
        "--tb=short",
        "--strict-markers",
        "-ra"  # Show all except passed
    ])
    
    print(f"📋 Running tests: {', '.join(files_to_run)}")
    print(f"🔧 Pytest args: {' '.join(pytest_args)}")
    print("-"*60)
    
    # Record start time
    start_time = time.time()
    
    # Run tests
    try:
        exit_code = pytest.main(pytest_args)
        success = exit_code == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        success = False
        exit_code = 1
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate summary
    result_summary = {
        "test_category": test_category,
        "files_tested": files_to_run,
        "success": success,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Category: {test_category}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Exit Code: {exit_code}")
    
    return result_summary


def run_ml_feature_validation() -> Dict:
    """
    Run comprehensive ML feature validation.
    
    Returns:
        Validation results
    """
    print("\n🔍 ML Feature Validation")
    print("="*60)
    
    validation_results = {
        "api_endpoints": False,
        "database_models": False,
        "service_layer": False,
        "workflows": False,
        "rag_compatibility": False,
        "overall": False
    }
    
    # Test categories in order of dependency
    test_sequence = [
        ("ml_models", "database_models"),
        ("ml_service", "service_layer"),
        ("ml_workflows", "workflows"),
        ("ml_api", "api_endpoints"),
        ("rag_regression", "rag_compatibility")
    ]
    
    overall_success = True
    
    for test_category, result_key in test_sequence:
        print(f"\n🧪 Testing {test_category}...")
        result = run_test_suite(test_category, verbose=False)
        
        validation_results[result_key] = result["success"]
        if not result["success"]:
            overall_success = False
            print(f"❌ {test_category} tests failed")
        else:
            print(f"✅ {test_category} tests passed")
    
    validation_results["overall"] = overall_success
    
    print("\n" + "="*60)
    print("🎯 VALIDATION SUMMARY")
    print("="*60)
    
    for component, status in validation_results.items():
        if component != "overall":
            status_icon = "✅" if status else "❌"
            print(f"{component.replace('_', ' ').title()}: {status_icon}")
    
    print(f"\nOverall Status: {'🎉 ALL PASSED' if overall_success else '⚠️ SOME FAILED'}")
    
    return validation_results


def generate_test_report(results: Dict) -> str:
    """
    Generate detailed test report.
    
    Args:
        results: Test results dictionary
    
    Returns:
        Path to generated report file
    """
    report_data = {
        "test_execution": results,
        "environment_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": os.getcwd()
        },
        "coverage_info": {
            "html_report": "htmlcov/index.html",
            "note": "Coverage report generated if running full test suite"
        }
    }
    
    # Generate report file
    report_file = Path("test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"📄 Test report generated: {report_file}")
    return str(report_file)


def check_test_prerequisites() -> bool:
    """
    Check if all test prerequisites are met.
    
    Returns:
        True if all prerequisites are met
    """
    print("🔍 Checking test prerequisites...")
    
    prerequisites_met = True
    
    # Check required packages
    required_packages = [
        "pytest", "pytest-asyncio", "fastapi", "pandas", 
        "scikit-learn", "sqlalchemy"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            prerequisites_met = False
    
    # Check test files exist
    test_files = [
        "test_ml_api.py", "test_ml_models.py", "test_ml_service.py",
        "test_ml_workflows.py", "test_rag_regression.py"
    ]
    
    test_dir = Path(__file__).parent
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"✅ {test_file} found")
        else:
            print(f"❌ {test_file} missing")
            prerequisites_met = False
    
    # Check app structure
    app_components = [
        "../app/main.py",
        "../app/models/ml_models.py",
        "../app/services/ml_pipeline_service.py",
        "../workflows/ml/algorithm_registry.py"
    ]
    
    for component in app_components:
        component_path = test_dir / component
        if component_path.exists():
            print(f"✅ {component} found")
        else:
            print(f"❌ {component} missing")
            prerequisites_met = False
    
    if prerequisites_met:
        print("🎉 All prerequisites met!")
    else:
        print("⚠️ Some prerequisites are missing. Tests may fail.")
    
    return prerequisites_met


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML Features Test Suite")
    parser.add_argument(
        "--category", 
        choices=["ml_api", "ml_models", "ml_service", "ml_workflows", "rag_regression", "existing", "all"],
        default="all",
        help="Category of tests to run"
    )
    parser.add_argument("--validate", action="store_true", help="Run comprehensive validation")
    parser.add_argument("--check-prereqs", action="store_true", help="Check prerequisites only")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    
    args = parser.parse_args()
    
    if args.check_prereqs:
        check_test_prerequisites()
        sys.exit(0)
    
    # Check prerequisites first
    if not check_test_prerequisites():
        print("\n⚠️ Prerequisites not met. Continuing anyway...")
    
    if args.validate:
        results = run_ml_feature_validation()
    else:
        results = run_test_suite(args.category, verbose=not args.quiet)
    
    if args.report:
        report_file = generate_test_report(results)
        print(f"Report saved to: {report_file}")
    
    # Exit with appropriate code
    if isinstance(results, dict):
        exit_code = 0 if results.get("success", False) or results.get("overall", False) else 1
    else:
        exit_code = 1
    
    sys.exit(exit_code)