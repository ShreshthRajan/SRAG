#!/usr/bin/env python3
"""
Comprehensive test of JSON cleaning fix.
Tests real-world malformed JSON that models produce.
"""

import json
import re

def clean_json_string(json_str: str) -> str:
    """Fixed JSON cleaning function."""
    try:
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')

        # Fix unquoted keys - only match keys that aren't already quoted
        json_str = re.sub(r'(?<!")(\b\w+\b)(?!")(\s*):', r'"\1"\2:', json_str)

        # Remove control characters
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')

        # Fix common typos
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')

        return json_str
    except Exception:
        return json_str


def test_comprehensive():
    """Test comprehensive scenarios."""

    test_cases = [
        # Basic valid JSON - should stay valid
        (
            '{"test_cases": [{"name": "test1", "input": [1, 2], "expected": 3}]}',
            "Valid nested JSON"
        ),
        # Unquoted keys
        (
            '{test_cases: [{name: "test1", input: [1, 2], expected: 3}]}',
            "Unquoted keys"
        ),
        # Trailing commas
        (
            '{"test_cases": [{"name": "test1", "input": [1, 2,], "expected": 3,},]}',
            "Multiple trailing commas"
        ),
        # Python booleans
        (
            '{"test_cases": [{"name": "test1", "valid": True, "expected": False}]}',
            "Python booleans"
        ),
        # Python None
        (
            '{"test_cases": [{"name": "test1", "expected": None}]}',
            "Python None"
        ),
        # Single quotes
        (
            "{'test_cases': [{'name': 'test1', 'input': [1, 2]}]}",
            "Single quotes"
        ),
        # Mixed issues (realistic model output)
        (
            "{test_cases: [{'name': 'Basic Test', input: [1, 2,], expected: True,}]}",
            "Multiple issues combined"
        ),
    ]

    print("COMPREHENSIVE JSON CLEANING TEST")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_input, description in test_cases:
        try:
            cleaned = clean_json_string(test_input)
            parsed = json.loads(cleaned)
            print(f"✅ {description}")
            passed += 1
        except json.JSONDecodeError as e:
            print(f"❌ {description}")
            print(f"   Error: {e}")
            print(f"   Input: {test_input[:80]}...")
            print(f"   Cleaned: {cleaned[:80]}...")
            failed += 1
        except Exception as e:
            print(f"❌ {description} - Unexpected error: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")

    if failed == 0:
        print("✅ ALL TESTS PASSED - JSON cleaning is production-ready")
        return 0
    else:
        print(f"❌ {failed} TESTS FAILED - JSON cleaning needs more work")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_comprehensive())
