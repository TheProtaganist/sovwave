# Ollama Algorithm Evolver V2 - Enhanced Features

## Major Improvements

### 1. Multiple Algorithms Per Round (2-16)
- **Configurable**: Set 2 to 16 algorithms per generation
- **Default**: 6 algorithms per round
- **UI Control**: "Algorithms/Round" field in GUI

```python
# Config
algorithms_per_round: int = 6  # 2-16 range
```

### 2. Hybrid Evolution Strategy

**50% Variants + 50% New Algorithms** (configurable)

**Variants** (n/2):
- Improved versions of current best
- Focus on weakest metrics
- Must be fundamentally different (not just parameter tweaks)

**New Algorithms** (n/2):
- Completely original approaches
- Not based on existing code
- Creative algorithmic diversity

```python
# Example: 6 algorithms per round
# → 3 variants + 3 new algorithms
variant_ratio: float = 0.5  # Adjustable
```

### 3. Code Difference Enforcement

**NO MORE LAZY PARAMETER CHANGES!**

```python
enforce_code_difference: bool = True
min_code_difference_ratio: float = 0.3  # At least 30% different
```

**How it works:**
1. Normalizes code (removes comments, whitespace)
2. Calculates character-level differences
3. Checks structural differences (keywords, patterns)
4. **Rejects** algorithms that are <30% different

**Example of REJECTED lazy change:**
```julia
# Original
for i in 1:n
    sum += i
end

# REJECTED (only parameter changed)
for i in 1:100  # Just changed n to 100!
    sum += i
end

# ACCEPTED (different approach)
sum = n * (n + 1) ÷ 2  # Gauss formula!
```

### 4. Auto-Calculate Expected Values

**No manual testing needed!**

```python
auto_calculate_expected: bool = True
validation_runs: int = 5
```

**Process:**
1. Runs initial code 5 times
2. Verifies results are consistent
3. Auto-extracts expected value
4. Uses for all subsequent benchmarks

**GUI:**
- Leave "Expected" field empty → auto-calculates
- Or provide manual expected value

**Example:**
```julia
function factorial(n::Int)::Int
    n <= 1 ? 1 : n * factorial(n-1)
end
```

Input: `5`
Auto-calculated: `120`

### 5. External Package Support

**Automatically extracts and preserves imports!**

```julia
using LinearAlgebra
using Statistics
using BenchmarkTools

function matrix_norm(A::Matrix{Float64})::Float64
    return norm(A)  # Uses LinearAlgebra
end
```

**Features:**
- Extracts `using` and `import` statements
- Tracks project packages
- Includes in all generated variants
- Validates package availability

### 6. Enhanced Prompt Engineering

**Variants Prompt:**
```
CRITICAL REQUIREMENTS:
- Use FUNDAMENTALLY DIFFERENT logic/algorithm
- NOT just parameter tweaks
- Different data structures, algorithms, loop orders
- Mathematical identities, vectorization
```

**New Algorithm Prompt:**
```
Create a COMPLETELY DIFFERENT algorithmic approach
DO NOT copy patterns from previous attempts
Examples:
- Iterative vs recursive vs matrix math
- Different loop structures, vectorization
- Memoization, dynamic programming
- Bit manipulation, algebraic shortcuts
```

### 7. Validation & Quality Control

**Multiple safeguards prevent bad algorithms:**

1. **Syntax validation** before benchmarking
2. **Minimum score threshold** (300/1000)
3. **Accuracy-first weighting** (30%)
4. **Code difference check** (30% minimum)
5. **Consistent expected values** (5 validation runs)

### 8. Improved Metrics Display

**Evolution tab shows:**
```
[Gen 2] fibonacci_Gen2_Variant1 - Score: 785.23 ([EXCELLENT])
  Acc: 100.0%, Spd: 85.2%, Cplx: 30.0%, Coh: 92.5%

[Gen 2] fibonacci_Gen2_New1 - Score: 812.45 ([EXCELLENT])
  Acc: 100.0%, Spd: 88.7%, Cplx: 25.0%, Coh: 95.0%

[REJECTED] Variant too similar to existing code
[IMPROVEMENT] fibonacci_Gen2_New1: 812.45
```

## Configuration Options

### GUI Controls

**Algorithms/Round**: 2-16 (default: 6)
**Rounds**: -1 for unlimited, or specific number
**Model**: Auto-detected from your Ollama models
**Input**: Test value
**Expected**: Leave empty for auto-calculation

### Checkboxes

☑ **Auto-calculate expected values**
- Runs initial code to determine expected output
- Validates consistency across 5 runs

☑ **Enforce code differences**
- Prevents lazy parameter tweaking
- Requires 30% structural difference

### Config Menu

**Edit Weights**:
- accuracy: 0.3
- speed: 0.2
- complexity: 0.1
- coherence: 0.15
- stability: 0.15
- memory: 0.1

## Usage Examples

### Example 1: Fibonacci Evolution

**Initial** (naive recursive O(2^n)):
```julia
function fibonacci(n::Int)::Int
    n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2)
end
```

**Generation 1 Output:**
- 3 variants (memoization, iterative, formula-based)
- 3 new approaches (matrix, lookup table, Binet's)

**Generation 2 Output:**
- 3 variants of best Gen1 (optimized matrix)
- 3 completely new (closed-form, generating function, etc.)

**Final Result**: O(log n) matrix exponentiation - [PERFECT] score

### Example 2: Sort Algorithm

**Initial** (bubble sort O(n²)):
```julia
function sort_array(arr::Vector{Int})::Vector{Int}
    n = length(arr)
    result = copy(arr)
    for i in 1:n
        for j in 1:n-i
            if result[j] > result[j+1]
                result[j], result[j+1] = result[j+1], result[j]
            end
        end
    end
    return result
end
```

**Input**: `[64, 34, 25, 12, 22, 11, 90]`
**Auto-calculated Expected**: `[11, 12, 22, 25, 34, 64, 90]`

**Evolution produces:**
- Quick sort variants
- Merge sort approaches
- Heap sort implementations
- Radix sort for integers
- Built-in sort! with optimizations

### Example 3: With External Packages

**Initial**:
```julia
using LinearAlgebra
using Statistics

function matrix_stats(A::Matrix{Float64})::Tuple{Float64, Float64, Float64}
    """Calculate mean, std, and norm of matrix"""
    m = mean(A)
    s = std(A)
    n = norm(A)
    return (m, s, n)
end
```

**Package handling:**
- Extracts: `using LinearAlgebra`, `using Statistics`
- Includes in all generated variants
- Validates packages are installed
- Offers to install if missing

## Architecture Changes

### Before (V1):
```
Generate 6 variants of current best
→ Pick best variant
→ Repeat
```

### After (V2):
```
Generate 3 variants of current best (must be 30% different)
+ Generate 3 completely new approaches
→ Track all codes to enforce diversity
→ Pick best from all 6
→ Repeat
```

## Code Difference Algorithm

```python
def calculate_code_difference(code1, code2):
    # 1. Normalize (remove comments, whitespace)
    norm1 = normalize(code1)
    norm2 = normalize(code2)
    
    # 2. Character-level difference (70% weight)
    char_diff = character_difference(norm1, norm2)
    
    # 3. Structural difference (30% weight)
    keywords1 = extract_keywords(code1)  # for, while, if, etc.
    keywords2 = extract_keywords(code2)
    struct_diff = keyword_difference(keywords1, keywords2)
    
    # 4. Weighted combination
    return (char_diff * 0.7) + (struct_diff * 0.3)
```

**Thresholds:**
- < 0.3: REJECTED (too similar)
- 0.3-0.5: Acceptable
- 0.5-0.8: Good diversity
- > 0.8: Completely different

## Performance Tips

### For 4B Models:
```
algorithms_per_round: 4  # 2 variants + 2 new
rounds: 10
temperature: 0.7
```

### For 7B Models:
```
algorithms_per_round: 6  # 3 variants + 3 new
rounds: 15
temperature: 0.7
```

### For 13B+ Models:
```
algorithms_per_round: 12  # 6 variants + 6 new
rounds: 20
temperature: 0.6
```

## Expected Performance

**Small functions** (< 20 lines):
- Convergence: 5-10 generations
- Total time: 5-15 minutes

**Medium functions** (20-50 lines):
- Convergence: 10-20 generations
- Total time: 15-30 minutes

**Complex functions** (50+ lines):
- Convergence: 20-40 generations
- Total time: 30-60 minutes

## Troubleshooting

### "All variants rejected as too similar"
- Increase `temperature` (more creativity)
- Decrease `min_code_difference_ratio`
- Use larger model (better at diversity)

### "Auto-calculation failed"
- Check initial code syntax
- Verify function returns consistent results
- Provide manual expected value

### "Package not found"
- GUI will offer to install missing packages
- Or manually: `julia -e 'using Pkg; Pkg.add("PackageName")'`

## Future Enhancements

Potential additions:
- [ ] Multi-objective optimization (Pareto front)
- [ ] Cross-breeding between algorithms
- [ ] Tournament selection with brackets
- [ ] Parallel variant generation
- [ ] GPU-accelerated benchmarking
- [ ] Export to Julia tournament file format

## Comparison: V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Algorithms/round | 6 (fixed) | 2-16 (configurable) |
| Algorithm types | Variants only | 50% variants + 50% new |
| Code diversity | No enforcement | 30% difference required |
| Expected values | Manual | Auto-calculated |
| Packages | Not supported | Full support |
| Lazy parameters | Allowed | Blocked |
| Prompts | Generic | Enforces creativity |

## Summary

V2 is a **complete overhaul** focused on:
1. **Diversity**: No more lazy parameter tweaking
2. **Automation**: Auto-calculates expected values
3. **Scalability**: 2-16 algorithms per round
4. **Quality**: Multiple validation layers
5. **Creativity**: Forces genuinely different approaches

The result: **Higher quality algorithms that truly explore the solution space!** 🌊🚀
