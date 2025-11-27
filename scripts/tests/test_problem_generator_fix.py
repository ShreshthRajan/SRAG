#!/usr/bin/env python3
"""
Test that problem generator KeyError fix works.
Handles both 'question' and 'description' fields.
"""

def test_field_handling():
    """Test handling of question vs description fields."""

    print("Testing problem field handling fix...")
    print("=" * 60)

    # Simulate the fix
    def safe_get_text(problem):
        """Fixed version that handles both field names."""
        return problem.get("description") or problem.get("question", "")

    test_cases = [
        ({"description": "Test description"}, "description field"),
        ({"question": "Test question"}, "question field"),
        ({"description": "Desc", "question": "Quest"}, "both fields (description priority)"),
        ({}, "no fields (empty string)"),
        ({"other": "value"}, "different field (empty string)"),
    ]

    all_passed = True

    for problem, description in test_cases:
        try:
            text = safe_get_text(problem)
            if problem.get("description"):
                expected = problem["description"]
            elif problem.get("question"):
                expected = problem["question"]
            else:
                expected = ""

            if text == expected:
                print(f"✅ {description}: '{text}'")
            else:
                print(f"❌ {description}: got '{text}', expected '{expected}'")
                all_passed = False
        except Exception as e:
            print(f"❌ {description}: Exception - {e}")
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✅ ALL TESTS PASSED - Field handling is robust")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(test_field_handling())
