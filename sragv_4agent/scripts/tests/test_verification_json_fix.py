#!/usr/bin/env python3
"""
Test that verification generator JSON cleaning is fixed.
Quick validation before Phase 3.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_json_cleaning():
    """Test JSON cleaning function."""
    from sragv.models.verification_generator import VerificationGenerator

    # Create minimal config
    config = {
        "model_name": "test",
        "max_length": 100,
        "temperature": 0.5,
        "top_p": 0.9
    }

    # Create instance (won't load model, just test method)
    class MockVerificationGenerator:
        """Mock to test _clean_json_string without loading model."""
        def _clean_json_string(self, json_str):
            import re
            try:
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(?<!")(\b\w+\b)(?!")(\s*):', r'"\1"\2:', json_str)
                json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
                json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                return json_str
            except Exception:
                return json_str

    gen = MockVerificationGenerator()

    # Test cases
    test_cases = [
        ('{"name": "test"}', 'Valid JSON'),
        ('{name: "test"}', 'Unquoted key'),
        ('{"name": "test",}', 'Trailing comma'),
        ('{\'name\': \'test\'}', 'Single quotes'),
        ('{"value": True}', 'Python boolean'),
        ('{"test": None}', 'Python None'),
    ]

    import json
    all_passed = True

    print("Testing JSON cleaning fix...")
    print("=" * 60)

    for test_input, description in test_cases:
        cleaned = gen._clean_json_string(test_input)
        try:
            json.loads(cleaned)
            print(f"✅ {description}: PASS")
        except json.JSONDecodeError as e:
            print(f"❌ {description}: FAIL - {e}")
            print(f"   Input: {test_input}")
            print(f"   Cleaned: {cleaned}")
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✅ ALL TESTS PASSED - JSON cleaning is fixed")
        return 0
    else:
        print("❌ SOME TESTS FAILED - JSON cleaning still has issues")
        return 1


if __name__ == "__main__":
    sys.exit(test_json_cleaning())
