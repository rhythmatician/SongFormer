"""
Quick test script for real-time inference
Tests audio device access and basic setup
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")

    try:
        import sounddevice as sd

        print("✓ sounddevice imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import sounddevice: {e}")
        print("  Please install: pip install sounddevice")
        return False

    try:
        import torch

        print(f"✓ PyTorch imported successfully (version {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False

    try:
        import librosa

        print(f"✓ librosa imported successfully (version {librosa.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import librosa: {e}")
        return False

    try:
        import numpy as np

        print(f"✓ numpy imported successfully (version {np.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False

    return True


def test_audio_devices():
    """Test audio device detection"""
    print("\nTesting audio devices...")

    try:
        import sounddevice as sd

        devices = sd.query_devices()

        # Find input devices
        input_device_indices = [
            i for i, d in enumerate(devices) if d["max_input_channels"] > 0
        ]

        if not input_device_indices:
            print("✗ No audio input devices found!")
            print("  Please connect a microphone or check your audio settings.")
            return False

        print(f"✓ Found {len(input_device_indices)} audio input device(s):")
        for device_idx in input_device_indices:
            device = devices[device_idx]
            default_marker = (
                " (default)" if device == sd.query_devices(kind="input") else ""
            )
            print(f"  [{device_idx}] {device['name']}{default_marker}")
            print(f"      Sample rate: {device['default_samplerate']} Hz")

        return True

    except Exception as e:
        print(f"✗ Error querying audio devices: {e}")
        return False


def test_model_files():
    """Test that required model files exist"""
    print("\nChecking model files...")

    required_paths = [
        "src/SongFormer/ckpts/MusicFM/msd_stats.json",
        "src/SongFormer/ckpts/MusicFM/pretrained_msd.pt",
        "src/SongFormer/ckpts/SongFormer.safetensors",
        "src/SongFormer/configs/SongFormer.yaml",
    ]

    all_found = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")
            all_found = False

    if not all_found:
        print("\n  Some model files are missing.")
        print("  Please run the download script:")
        print("    cd src/SongFormer")
        print("    python utils/fetch_pretrained.py")

    return all_found


def test_simple_audio_capture():
    """Test simple audio capture"""
    print("\nTesting audio capture (2 seconds)...")

    try:
        import sounddevice as sd
        import numpy as np

        duration = 2  # seconds
        sample_rate = 24000

        print("  Recording...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        # Check if we got valid audio data
        if recording is not None and len(recording) > 0:
            max_amplitude = np.abs(recording).max()
            print(f"✓ Successfully captured {len(recording)} samples")
            print(f"  Max amplitude: {max_amplitude:.4f}")

            if max_amplitude < 0.001:
                print("  ⚠ Warning: Audio level is very low. Check your microphone.")

            return True
        else:
            print("✗ Failed to capture audio")
            return False

    except Exception as e:
        print(f"✗ Error during audio capture: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("SongFormer Real-time Inference - Setup Test")
    print("=" * 60)

    tests = [
        ("Import test", test_imports),
        ("Audio devices test", test_audio_devices),
        ("Model files test", test_model_files),
        ("Audio capture test", test_simple_audio_capture),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to run real-time inference.")
        print("\nTo get started:")
        print("  python realtime_inference.py --list-devices")
        print("  python realtime_inference.py --device 0")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
