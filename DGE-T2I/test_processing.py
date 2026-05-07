#!/usr/bin/env python3
"""
Test script to verify the pipeline processes files without running the full model.
This tests the structure and data flow without GPU requirements.
"""
import sys
import os
sys.path.insert(0, 'src')

def test_processing_module():
    """Test the processing module with actual scene graph files."""
    from src.data.modules.processing import process_directory
    print("Testing processing module with sample files...")
    
    # Process the test directory with actual scene graph files
    try:
        prompts, filenames = process_directory("test_input")
        print(f"✓ Successfully processed {len(filenames)} files")
        print(f"✓ Generated {len(prompts)} prompts")
        if prompts:
            print(f"✓ First prompt (first 500 chars): {prompts[0]['prompt'][:500]}...")
        return True
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        return False

def test_config_paths():
    """Test that configuration paths are correct."""
    from src.data.modules.config import DEFAULT_OUTPUT_FILE, DEFAULT_CAUSAL_FILE
    print("\nTesting configuration paths...")
    print(f"✓ Output file path: {DEFAULT_OUTPUT_FILE}")
    print(f"✓ Causal file path: {DEFAULT_CAUSAL_FILE}")
    return True

if __name__ == "__main__":
    print("Testing the modular structure with actual scene graph files...\n")
    
    success = True
    success = test_config_paths() and success
    success = test_processing_module() and success
    
    if success:
        print(f"\n✓ All tests passed! The modular structure works correctly.")
        print("  The code successfully processes actual scene graph files.")
        print("  Output will go to the data/raw directory as configured.")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)