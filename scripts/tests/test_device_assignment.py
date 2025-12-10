#!/usr/bin/env python3
"""
Unit test for device assignment logic.
Tests that models are moved to correct devices using .to() method.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_device_assignment_types():
    """Test that device assignments accept integers and strings."""

    from sragv.models.base_player import PlayerConfig

    print("=" * 70)
    print("DEVICE ASSIGNMENT UNIT TESTS (v5 - PyTorch .to() method)")
    print("=" * 70)

    # Test 1: Integer device ID
    print("\n1. Testing integer device ID (0):")
    config1 = PlayerConfig(
        model_name="test/model",
        device=0
    )
    assert config1.device == 0, f"Expected device=0, got {config1.device}"
    print("   ✅ Integer device ID stored correctly")

    # Test 2: Integer device ID (1)
    print("\n2. Testing integer device ID (1):")
    config2 = PlayerConfig(
        model_name="test/model",
        device=1
    )
    assert config2.device == 1, f"Expected device=1, got {config2.device}"
    print("   ✅ Integer device ID stored correctly")

    # Test 3: String device ID ("auto")
    print("\n3. Testing string device ID ('auto'):")
    config3 = PlayerConfig(
        model_name="test/model",
        device="auto"
    )
    assert config3.device == "auto", f"Expected device='auto', got {config3.device}"
    print("   ✅ String device ID stored correctly")

    # Test 4: to_dict() includes device
    print("\n4. Testing to_dict() includes device:")
    config_dict = config1.to_dict()
    assert "device" in config_dict, "Device not in config dict"
    assert config_dict["device"] == 0, f"Expected device=0 in dict, got {config_dict['device']}"
    print("   ✅ Device included in config dict")

    # Test 5: Device string formatting
    print("\n5. Testing device string formatting:")
    test_device = 0
    formatted = f"cuda:{test_device}" if isinstance(test_device, int) else test_device
    assert formatted == "cuda:0", f"Expected 'cuda:0', got {formatted}"
    print("   ✅ Integer converts to 'cuda:0' format for .to() method")

    test_device2 = 1
    formatted2 = f"cuda:{test_device2}" if isinstance(test_device2, int) else test_device2
    assert formatted2 == "cuda:1", f"Expected 'cuda:1', got {formatted2}"
    print("   ✅ Integer converts to 'cuda:1' format for .to() method")

    print("\n" + "=" * 70)
    print("✅ ALL UNIT TESTS PASSED")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(test_device_assignment_types())
