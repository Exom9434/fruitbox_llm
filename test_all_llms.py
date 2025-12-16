# test_all_llms.py
"""
Comprehensive test script to verify all LLM providers work correctly
with the existing Pydantic models and API configurations.
"""

import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Import your existing modules
from llm_client import call_llm
from llm_strategy import (
    Moves, MovesWithCoT, MovesWithPerStepReason, JudgeVerdict,
    SingleMoveWrapper, SingleMoveWithReasonWrapper, SimpleMoves,
    MoveObject, MoveObjectWithReason, SingleMove
)
from config import MODEL_CONFIG

class LLMTestResults:
    """Class to store and manage test results for all LLMs"""

    def __init__(self):
        self.results = {}
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def add_result(self, model_family: str, test_name: str, success: bool,
                   details: Dict[str, Any], error: Optional[str] = None):
        """Add a test result for a specific model and test"""
        if model_family not in self.results:
            self.results[model_family] = {}

        self.results[model_family][test_name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

    def print_summary(self):
        """Print a formatted summary of all test results"""
        print(f"\n{'='*60}")
        print(f"LLM TEST SUMMARY - {self.test_timestamp}")
        print(f"{'='*60}")

        for model_family, tests in self.results.items():
            print(f"\n🤖 {model_family.upper()}")
            print("-" * 30)

            total_tests = len(tests)
            passed_tests = sum(1 for test in tests.values() if test["success"])

            for test_name, result in tests.items():
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"  {test_name:25} {status}")
                if not result["success"] and result["error"]:
                    print(f"    Error: {result['error']}")

            print(f"\n  Overall: {passed_tests}/{total_tests} tests passed")

        print(f"\n{'='*60}")

    def save_detailed_report(self, filename: Optional[str] = None):
        """Save detailed test results to JSON file"""
        if filename is None:
            filename = f"llm_test_report_{self.test_timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"📄 Detailed report saved to: {filename}")


def create_test_board_state():
    """Create a simple test board state for testing"""
    return [
        [1, 2, 1, 3, 2],
        [3, 1, 2, 1, 3],
        [2, 3, 1, 2, 1],
        [1, 2, 3, 1, 2],
        [3, 1, 2, 3, 1]
    ]


def create_simple_test_prompt():
    """Create a simple test prompt for basic connectivity"""
    return """You are a game AI. Analyze the given board and find optimal moves.

Return your response as valid JSON with the following structure:
{
  "target_commands": [
    {
      "id": 1,
      "move": [[row1, col1], [row2, col2]]
    }
  ]
}

Find moves where you can connect matching numbers horizontally or vertically."""


def test_basic_connectivity(model_family: str, test_results: LLMTestResults):
    """Test basic API connectivity and JSON response"""
    print(f"  Testing basic connectivity...")

    try:
        board_state = create_test_board_state()

        # Test with a simple prompt override (no file needed)
        system_prompt, user_prompt, raw_response, parsed_result, history = call_llm(
            model_family=model_family,
            board_state=board_state,
            prompt_path=None,
            response_model=Moves,
            full_prompt_override=create_simple_test_prompt()
        )

        # Check if we got a response
        if not raw_response or "API Call Failed" in str(raw_response):
            test_results.add_result(
                model_family, "basic_connectivity", False,
                {"raw_response": raw_response},
                "API call failed or returned empty response"
            )
            return False

        # Check if response is valid JSON
        try:
            json_data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            test_results.add_result(
                model_family, "basic_connectivity", False,
                {"raw_response": raw_response},
                f"Invalid JSON response: {e}"
            )
            return False

        test_results.add_result(
            model_family, "basic_connectivity", True,
            {
                "response_length": len(raw_response),
                "parsed_moves_count": len(parsed_result) if parsed_result else 0,
                "json_structure": "target_commands" in json_data
            }
        )
        return True

    except Exception as e:
        test_results.add_result(
            model_family, "basic_connectivity", False,
            {"exception": str(e)},
            f"Exception during basic connectivity test: {e}"
        )
        return False


def test_pydantic_model(model_family: str, model_class, test_name: str, test_results: LLMTestResults):
    """Test a specific Pydantic model with the LLM"""
    print(f"  Testing {test_name}...")

    try:
        board_state = create_test_board_state()

        # Create model-specific prompt
        schema_info = model_class.model_json_schema()
        prompt = f"""{create_simple_test_prompt()}

IMPORTANT: Your response must match this exact schema:
{json.dumps(schema_info, indent=2)}"""

        system_prompt, user_prompt, raw_response, parsed_result, history = call_llm(
            model_family=model_family,
            board_state=board_state,
            prompt_path=None,
            response_model=model_class,
            full_prompt_override=prompt
        )

        if not raw_response or "API Call Failed" in str(raw_response):
            test_results.add_result(
                model_family, test_name, False,
                {"model_class": model_class.__name__},
                "API call failed"
            )
            return False

        # Try to validate with Pydantic
        try:
            validated_model = model_class.model_validate_json(raw_response)
            test_results.add_result(
                model_family, test_name, True,
                {
                    "model_class": model_class.__name__,
                    "validated": True,
                    "parsed_result_type": type(parsed_result).__name__,
                    "response_length": len(raw_response)
                }
            )
            return True

        except Exception as validation_error:
            test_results.add_result(
                model_family, test_name, False,
                {
                    "model_class": model_class.__name__,
                    "raw_response": raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
                },
                f"Pydantic validation failed: {validation_error}"
            )
            return False

    except Exception as e:
        test_results.add_result(
            model_family, test_name, False,
            {"model_class": model_class.__name__},
            f"Exception during {test_name} test: {e}"
        )
        return False


def test_all_models_for_llm(model_family: str, test_results: LLMTestResults):
    """Run all Pydantic model tests for a specific LLM"""
    print(f"\n🔍 Testing {model_family.upper()}...")

    # Test basic connectivity first
    if not test_basic_connectivity(model_family, test_results):
        print(f"  ❌ Basic connectivity failed for {model_family}")
        return

    # Test all Pydantic models
    models_to_test = [
        (Moves, "moves_model"),
        (MovesWithCoT, "moves_with_cot"),
        (MovesWithPerStepReason, "moves_with_reason"),
        (JudgeVerdict, "judge_verdict"),
        (SimpleMoves, "simple_moves"),
        (SingleMove, "single_move"),
        (SingleMoveWrapper, "single_move_wrapper"),
    ]

    for model_class, test_name in models_to_test:
        test_pydantic_model(model_family, model_class, test_name, test_results)


def check_api_keys():
    """Check if API keys are configured"""
    print("🔑 Checking API key configuration...")

    missing_keys = []

    for model_family, config in MODEL_CONFIG.items():
        api_key = config.get("api_key")

        # Special case for Ollama (uses dummy key)
        if model_family == "llama":
            if api_key and api_key != "":
                print(f"  ✅ {model_family}: Configured for local Ollama")
            else:
                print(f"  ⚠️  {model_family}: Not configured")
                missing_keys.append(model_family)
        else:
            # For other APIs, check for real API keys
            if not api_key or api_key in ["", "your_api_key_here", "None"]:
                missing_keys.append(model_family)
            else:
                print(f"  ✅ {model_family}: API key configured")

    if missing_keys:
        print(f"  ⚠️  Missing API keys for: {', '.join(missing_keys)}")
        return missing_keys

    return []


def main():
    """Main test execution function"""
    print("🚀 Starting LLM Multi-Provider Test Suite")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check API keys
    missing_keys = check_api_keys()

    if missing_keys:
        print(f"\n💡 To use missing providers, add your API keys to .env file:")
        for key in missing_keys:
            if key == "llama":
                print(f"   - For {key}: Install and run Ollama locally")
            else:
                env_var = f"{key.upper()}_API_KEY" if key != "gpt" else "OPENAI_API_KEY"
                if key == "claude":
                    env_var = "ANTHROPIC_API_KEY"
                elif key == "gemini":
                    env_var = "GOOGLE_API_KEY"
                print(f"   - For {key}: Set {env_var} in your .env file")

    # Initialize test results
    test_results = LLMTestResults()

    # Get available models (skip those with missing API keys)
    available_models = [
        model for model in MODEL_CONFIG.keys()
        if model not in missing_keys
    ]

    if not available_models:
        print("❌ No models available for testing (missing API keys)")
        return

    print(f"\n📝 Testing {len(available_models)} model(s): {', '.join(available_models)}")

    # Test each available model
    for model_family in available_models:
        try:
            test_all_models_for_llm(model_family, test_results)
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Critical error testing {model_family}: {e}")
            traceback.print_exc()

    # Print summary and save report
    test_results.print_summary()
    test_results.save_detailed_report()

    print(f"\n🎉 Test suite completed!")


if __name__ == "__main__":
    main()