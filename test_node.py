#!/usr/bin/env python3
"""
Test script for ACE-Step ComfyUI Node
"""

import os
import sys
import json
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock folder_paths for standalone testing
class MockFolderPaths:
    __file__ = __file__
    @staticmethod
    def get_output_directory():
        return tempfile.gettempdir()

# Replace folder_paths before importing nodes
sys.modules['folder_paths'] = MockFolderPaths()

# Now import the node
from nodes import ACE_STEP_TEXT_TO_MUSIC

def test_text_to_music():
    """Test the Text to Music node"""
    print("=" * 60)
    print("Testing ACE-Step Text to Music Node")
    print("=" * 60)

    # Model configuration - use the acestep_project with checkpoints symlink
    checkpoint_dir = "/Users/xiohu/work/ai-tools/models/Ace-Step1.5"
    config_path = "acestep-v15-turbo"
    lm_model_path = "acestep-5Hz-lm-1.7B"

    # Test parameters
    params = {
        "caption": "upbeat electronic dance music with heavy bass",
        "checkpoint_dir": checkpoint_dir,
        "config_path": config_path,
        "lm_model_path": lm_model_path,
        "duration": 20.0,  # Short duration for testing
        "batch_size": 1,
        "seed": 42,
        "inference_steps": 4,  # Low steps for faster testing
        "device": "mps",  # Use MPS for Mac M4
    }

    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Create node instance
    node = ACE_STEP_TEXT_TO_MUSIC()

    print("\nInitializing handlers...")
    try:
        dit_handler, llm_handler = node.initialize_handlers(
            checkpoint_dir=params["checkpoint_dir"],
            config_path=params["config_path"],
            lm_model_path=params["lm_model_path"],
            device=params["device"],
        )
        print("✓ Handlers initialized successfully!")
    except Exception as e:
        print(f"✗ Handler initialization failed: {e}")
        return False

    print("\nGenerating music...")
    try:
        audio_output, audio_path, metadata = node.generate(**params)

        print("✓ Generation successful!")
        print(f"\nResults:")
        print(f"  Audio shape: {audio_output['waveform'].shape}")
        print(f"  Sample rate: {audio_output['sample_rate']} Hz")
        print(f"  Audio path: {audio_path}")
        print(f"  Duration: {audio_output['waveform'].shape[1] / audio_output['sample_rate']:.2f} seconds")

        print(f"\nMetadata:")
        print(json.dumps(json.loads(metadata), indent=2))

        # Check if audio file exists
        if os.path.exists(audio_path):
            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"\n✓ Audio file created: {size_mb:.2f} MB")
        else:
            print(f"\n✗ Audio file not found at: {audio_path}")

        return True

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_text_to_music()
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED ✓")
    else:
        print("TEST FAILED ✗")
    print("=" * 60)
    sys.exit(0 if success else 1)
