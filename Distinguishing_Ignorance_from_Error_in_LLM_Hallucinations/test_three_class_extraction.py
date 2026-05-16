#!/usr/bin/env python3
"""
Test script to verify three-class hidden state extraction
Tests ModelInside with three-class data path configuration
"""
import os
import sys

# Test 1: Verify ModelInside can be imported
print("=" * 60)
print("Test 1: Importing ModelInside...")
try:
    from ModelInside import ModelInside
    print("✓ ModelInside imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ModelInside: {e}")
    sys.exit(1)

# Test 2: Verify __init__ accepts data_path_general parameter
print("\n" + "=" * 60)
print("Test 2: Checking ModelInside.__init__ signature...")
import inspect
sig = inspect.signature(ModelInside.__init__)
params = list(sig.parameters.keys())
if 'data_path_general' in params:
    print(f"✓ data_path_general parameter found in __init__")
    print(f"  Parameters: {params}")
else:
    print(f"✗ data_path_general parameter NOT found in __init__")
    print(f"  Parameters: {params}")
    sys.exit(1)

# Test 3: Verify generate_data_three_classes method exists
print("\n" + "=" * 60)
print("Test 3: Checking generate_data_three_classes method...")
if hasattr(ModelInside, 'generate_data_three_classes'):
    print("✓ generate_data_three_classes method found")
else:
    print("✗ generate_data_three_classes method NOT found")
    sys.exit(1)

# Test 4: Verify save_all_data_three_classes method exists
print("\n" + "=" * 60)
print("Test 4: Checking save_all_data_three_classes method...")
if hasattr(ModelInside, 'save_all_data_three_classes'):
    print("✓ save_all_data_three_classes method found")
else:
    print("✗ save_all_data_three_classes method NOT found")
    sys.exit(1)

# Test 5: Verify dataset files exist
print("\n" + "=" * 60)
print("Test 5: Checking dataset files...")
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name_safe = model_name.replace('/', '_')
dataset_path = "datasets/"

files_needed = [
    f"{dataset_path}HallucinateTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
    f"{dataset_path}NonHallucinateTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
    f"{dataset_path}GeneralTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
]

all_exist = True
for file_path in files_needed:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✓ {os.path.basename(file_path)} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {file_path} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n⚠ Some dataset files are missing!")

# Test 6: Instantiate ModelInside with three-class config (dry run)
print("\n" + "=" * 60)
print("Test 6: Creating ModelInside instance with three-class config...")
try:
    model_instance = ModelInside(
        path_to_save_results="test_results/",
        data_path_without_hallucinations=f"{dataset_path}NonHallucinateTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
        data_path_with_hallucinations=f"{dataset_path}HallucinateTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
        data_path_general=f"{dataset_path}GeneralTrivia_qa_no_contextWithThreshold1.0_{model_name_safe}.json",
        model_name=model_name,
        dataset_size=10,  # Small size for testing
        dataset_name="trivia_qa",
        threshold_of_data=1.0,
        concat_answer=False,
        static_dataset=False,
        alice=False
    )
    print("✓ ModelInside instance created successfully")
    print(f"  Model: {model_instance.MODEL_NAME}")
    print(f"  Dataset size: {model_instance.dataset_size}")
    print(f"  Save path: {model_instance.path_to_save_results}")
    print(f"  Has data_path_general: {model_instance.data_path_general is not None}")
except Exception as e:
    print(f"✗ Failed to create ModelInside instance: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All basic tests passed! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python RunAllSteps.py --run_initial_test --model_name meta-llama/Llama-3.2-1B-Instruct")
print("2. Check for .npy files in results/ directory")
print("3. Verify 12 files are created (4 types × 3 classes)")
