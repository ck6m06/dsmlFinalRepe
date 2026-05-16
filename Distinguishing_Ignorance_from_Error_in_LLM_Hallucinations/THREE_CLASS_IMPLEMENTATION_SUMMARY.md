# Three-Class Hidden State Extraction - Implementation Summary

## Overview
Successfully implemented three-class simultaneous hidden state extraction for the hallucination mitigation project. This enhancement allows extraction of representations for Hallucinate, NonHallucinate, and General classes in a single unified pipeline, eliminating redundant model loading and forward passes.

## Changes Made

### 1. ModelInside.py - Core Changes

#### A. Modified `__init__()` (Line 24)
**Before:**
```python
def __init__(self, path_to_save_results, data_path_without_hallucinations, data_path_with_hallucinations,
             model_name="GOAT-AI/GOAT-7B-Community", dataset_size=1000, ...):
```

**After:**
```python
def __init__(self, path_to_save_results, data_path_without_hallucinations, data_path_with_hallucinations,
             model_name="GOAT-AI/GOAT-7B-Community", dataset_size=1000, ..., data_path_general=None):
```

**Added Line 37:**
```python
self.data_path_general = data_path_general
```

**Key Feature:** Optional parameter allows backward compatibility - if `data_path_general=None`, uses original two-class mode.

#### B. Modified `generate_data()` (Lines 49-56)
**New Logic:**
```python
# Check if we have three datasets (general classification mode)
if self.data_path_general is not None:
    return self.generate_data_three_classes()
```

**Behavior:**
- If `data_path_general` is provided, uses new three-class extraction
- If `data_path_general` is None, uses original two-class extraction
- Maintains backward compatibility with existing code

#### C. Added `generate_data_three_classes()` Method (Lines 89-148)
**Purpose:** Simultaneous extraction of hidden states for all three classes

**Implementation:**
1. Loads all three datasets simultaneously
2. Extracts representations using single model instance with three separate calls:
   - `generate_text_func(data_with_hallucinations, tag='hallucinate', ...)`
   - `generate_text_func(data_without_hallucinations, tag='nonhallucinate', ...)`
   - `generate_text_func(data_general, tag='general', ...)`
3. Sets `random.seed(42)` before each extraction for reproducibility
4. Passes `static=self.static_dataset` to maintain parameter consistency
5. Calls `save_all_data_three_classes()` to save 12 files

**Return Value:**
- Tuple of 12 numpy arrays (4 types × 3 classes)

#### D. Added `save_all_data_three_classes()` Method (Lines 191-210)
**Purpose:** Save all 12 hidden state files with class-specific naming

**Output Files (per configuration):**
```
_all_mlp_vector_hallucinate.npy
_all_attention_vector_hallucinate.npy
_heads_vectors_hallucinate_no_projection.npy
_all_residual_vectors_hallucinate.npy

_all_mlp_vector_nonhallucinate.npy
_all_attention_vector_nonhallucinate.npy
_heads_vectors_nonhallucinate_no_projection.npy
_all_residual_vectors_nonhallucinate.npy

_all_mlp_vector_general.npy
_all_attention_vector_general.npy
_heads_vectors_general_no_projection.npy
_all_residual_vectors_general.npy
```

### 2. RunAllSteps.py - Integration Changes

#### Modified `run_initial_test_on_dataset()` Function

**Change 1: Three-class extraction without concatenation (Lines 95-110)**
```python
MLPCheck = ModelInside("results/",
                       data_path_without_hallucinations=path_without,
                       data_path_with_hallucinations=path_with,
                       data_path_general=path_general,  # NEW PARAMETER
                       ...)
(all_mlp_vector_hall, all_attention_vector_hall, 
 all_mlp_vector_nonhall, all_attention_vector_nonhall, 
 all_mlp_vector_gen, all_attention_vector_gen,
 heads_vectors_hall, heads_vectors_nonhall, heads_vectors_gen,
 all_residual_hall, all_residual_nonhall, all_residual_gen) = MLPCheck.generate_data()
```

**Change 2: Three-class extraction with concatenation (Lines 127-140)**
```python
MLPCheck = ModelInside("results/",
                       data_path_without_hallucinations=path_without,
                       data_path_with_hallucinations=path_with,
                       data_path_general=path_general,  # NEW PARAMETER
                       concat_answer=True,
                       ...)
(all_mlp_vector_hall_concat, ...) = MLPCheck.generate_data()
```

**Benefits:**
- Single model instance instead of multiple separate instances
- Eliminates redundant model loading and CUDA memory allocation
- Consistent seed (42) across all three classes
- Clear separation of concerns: static vs. non-static, with vs. without concatenation

## Data Processing Flow

### Previous Pipeline (Two-Class, Separate Runs)
```
ModelInside Instance 1
├─ Load Model (meta-llama/Llama-3.2-1B-Instruct)
├─ Load Hallucinate Dataset
├─ Extract Hallucinate representations
├─ Load NonHallucinate Dataset
├─ Extract NonHallucinate representations
└─ Save 8 .npy files

ModelInside Instance 2  [REDUNDANT - Same Model Loaded Again]
├─ Load Model (meta-llama/Llama-3.2-1B-Instruct)
├─ Load Hallucinate Dataset
├─ Extract Hallucinate representations
├─ Load NonHallucinate Dataset
├─ Extract NonHallucinate representations
└─ Save 8 More .npy Files
```

### New Pipeline (Three-Class, Unified Run)
```
ModelInside Instance
├─ Load Model (meta-llama/Llama-3.2-1B-Instruct) [ONCE]
├─ Load All 3 Datasets
│  ├─ Hallucinate Dataset
│  ├─ NonHallucinate Dataset
│  └─ General Dataset
├─ Extract All Representations [3 separate calls, same model]
│  ├─ Hallucinate class
│  ├─ NonHallucinate class
│  └─ General class
└─ Save 12 .npy Files (4 types × 3 classes)
```

## Key Features

### 1. Reproducibility
- `random.seed(42)` set before each class extraction
- Ensures consistent results across runs
- Maintains deterministic behavior

### 2. Backward Compatibility
- Original two-class mode preserved when `data_path_general=None`
- No breaking changes to existing code
- Existing scripts continue to work without modification

### 3. Efficiency
- Single model load vs. multiple loads
- Single forward pass per class vs. redundant passes
- Reduced memory footprint on GPU

### 4. Extensibility
- Method `save_all_data_three_classes()` clearly separated
- Easy to add more classes in future
- Clean parameter naming with class suffixes

## Testing & Validation

### File Integrity
✓ No syntax errors in modified files
✓ All class methods properly defined
✓ Function signatures consistent

### Dataset Verification
✓ Three dataset files exist for meta-llama/Llama-3.2-1B-Instruct:
  - HallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json
  - NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json
  - GeneralTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json

## Usage Examples

### Two-Class Mode (Original Behavior)
```python
from ModelInside import ModelInside

model = ModelInside(
    path_to_save_results="results/",
    data_path_without_hallucinations="path/to/nonhallucinate.json",
    data_path_with_hallucinations="path/to/hallucinate.json",
    model_name="meta-llama/Llama-3.2-1B-Instruct"
)
results = model.generate_data()  # Returns 8 vectors
```

### Three-Class Mode (New Feature)
```python
from ModelInside import ModelInside

model = ModelInside(
    path_to_save_results="results/",
    data_path_without_hallucinations="path/to/nonhallucinate.json",
    data_path_with_hallucinations="path/to/hallucinate.json",
    data_path_general="path/to/general.json",  # NEW
    model_name="meta-llama/Llama-3.2-1B-Instruct"
)
results = model.generate_data()  # Returns 12 vectors
```

### Via RunAllSteps
```bash
python RunAllSteps.py \
    --run_initial_test \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name trivia_qa \
    --threshold 1.0 \
    --dataset_size 1000
```

## Output Structure

Generated files follow pattern: `results/<model>/<dataset>/<threshold>/concat_answer<bool>_size<N>/`

### Three-Class Output (12 files total)
```
results/meta-llama_Llama-3.2-1B-Instruct/trivia_qa/1.0/concat_answerFalse_size1000/
├── _all_mlp_vector_hallucinate.npy
├── _all_attention_vector_hallucinate.npy
├── _heads_vectors_hallucinate_no_projection.npy
├── _all_residual_vectors_hallucinate.npy
├── _all_mlp_vector_nonhallucinate.npy
├── _all_attention_vector_nonhallucinate.npy
├── _heads_vectors_nonhallucinate_no_projection.npy
├── _all_residual_vectors_nonhallucinate.npy
├── _all_mlp_vector_general.npy
├── _all_attention_vector_general.npy
├── _heads_vectors_general_no_projection.npy
└── _all_residual_vectors_general.npy
```

## Next Steps

1. **Run three-class extraction:**
   ```bash
   cd d:\DSML2\hallucination-mitigation\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations
   python RunAllSteps.py --run_initial_test --model_name meta-llama/Llama-3.2-1B-Instruct
   ```

2. **Verify output:**
   - Check for 12 .npy files in results directory
   - Verify file sizes are reasonable (typical: 50-500 MB each)
   - Confirm consistent shapes across classes

3. **Run analysis:**
   ```bash
   python plot_results.py
   ```
   - Use existing `linear_classifier_3_options()` for three-class analysis
   - Plot confusion matrices and accuracy metrics
   - Compare against baseline two-class results

## Summary

The three-class implementation is complete and ready for testing. The changes:
- ✅ Maintain backward compatibility
- ✅ Reduce redundant computation
- ✅ Improve memory efficiency
- ✅ Align with existing plot_results.py three-class analysis capability
- ✅ Support multiple LLM models and datasets
- ✅ Follow existing code patterns and conventions

**Status: Implementation Complete - Ready for Testing**
