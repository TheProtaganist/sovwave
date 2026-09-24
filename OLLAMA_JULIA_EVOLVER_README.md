# Ollama Julia Algorithm Evolution Engine

AI-powered Julia algorithm optimizer using small LLMs (4B-7B models) to evolve optimal code through competitive rounds.

## Features

### Core Capabilities
- **Julia-native benchmarking** using BenchmarkTools.jl
- **Multi-round evolution** with tournament-style competition
- **Comprehensive metrics**: accuracy, speed, complexity, coherence, stability, memory
- **Automatic convergence detection** - finds optimal peak
- **Winners tracking** in markdown files
- **Smart model integration** - auto-detects your Ollama models
- **Prevents bad code** - validation + minimum score thresholds

### Metrics (Path of Least Resistance)
- **[PERFECT]** (Score ≥ 950): Flawless, mathematically exact, optimal
- **[EXCELLENT]** (Score ≥ 750): Great approximation, minimal tradeoffs  
- **[COMPROMISE]** (Score ≥ 500): Acceptable tradeoffs
- **[BROKEN]** (Score < 500): Discarded

## Requirements

### Software
```bash
# 1. Julia (required)
# Download from: https://julialang.org/downloads/

# 2. BenchmarkTools.jl (auto-installed by GUI)
julia -e 'using Pkg; Pkg.add("BenchmarkTools")'

# 3. Ollama (required)
# Install from: https://ollama.ai/

# 4. A coding model (4B-7B recommended for 6GB CUDA)
ollama pull qwen2.5-coder:7b
# OR
ollama pull qwen2.5-coder:4b
ollama pull deepseek-coder:6.7b
ollama pull codellama:7b
```

### Python Packages
```bash
pip3 install tkinter  # Usually included with Python
```

## Usage

### 1. Start Ollama
```bash
ollama serve
```

### 2. Launch GUI
```bash
python3 ollama_algorithm_evolver_gui.py
```

### 3. Configure Evolution
- **Function Name**: Enter Julia function name (e.g., `fibonacci`)
- **Rounds**: Number of evolution rounds (-1 for unlimited)
- **Model**: Select from your installed Ollama models
- **Test Data**: 
  - Input: Test value (e.g., `10`)
  - Expected: Expected output (e.g., `55` for fibonacci(10))

### 4. Start Evolution
1. Load or write initial Julia code
2. Click "Start Evolution"
3. Watch as the AI evolves better algorithms
4. Best code auto-saves to markdown

## Example: Fibonacci Evolution

**Initial Code** (naive recursive):
```julia
function fibonacci(n::Int)::Int
    if n <= 1
        return n
    end
    return fibonacci(n-1) + fibonacci(n-2)
end
```

**After Evolution** (optimized iterative):
```julia
function fibonacci(n::Int)::Int
    """Optimized Fibonacci using dynamic programming"""
    if n <= 1
        return n
    end
    
    a, b = 0, 1
    @inbounds for i in 2:n
        a, b = b, a + b
    end
    return b
end
```

**Potential Final** (matrix exponentiation):
```julia
function fibonacci(n::Int)::Int
    """Ultra-fast Fibonacci using matrix exponentiation O(log n)"""
    if n <= 1
        return n
    end
    
    function matrix_pow(base, exp)
        if exp == 1
            return base
        end
        if exp % 2 == 0
            half = matrix_pow(base, exp ÷ 2)
            return half * half
        else
            return base * matrix_pow(base, exp - 1)
        end
    end
    
    M = [1 1; 1 0]
    result = matrix_pow(M, n)
    return result[1, 2]
end
```

## How It Works

### Evolution Process
1. **Initial Evaluation**: Benchmark starting algorithm
2. **Weakness Detection**: Identify bottom 3 metrics
3. **AI Improvement**: Ollama generates 6 variants focusing on weak areas
4. **Validation**: Test accuracy, speed, and correctness
5. **Selection**: Keep only algorithms above minimum threshold
6. **Convergence Check**: Stop when optimal peak is found
7. **Winner Tracking**: Save best to `algorithm_evolution_winners.md`

### Metrics Calculation

**Accuracy** (0-1): Correctness over 100 test runs
```julia
accuracy = correct_runs / 100
```

**Speed** (0-1): Normalized execution time
```julia
speed = 1.0 / (1.0 + mean_time_ns / 1e6)
```

**Complexity** (0-1): Estimated from nested loops
- O(1): 0.1
- O(n): 0.3  
- O(n²): 0.6
- O(n³+): 0.9

**Coherence** (0-1): Code quality score
- Type annotations: +5%
- Docstrings: +10%
- Optimization macros (@inbounds, @simd): +10%
- Long lines: -20%
- Deep nesting: -10%

**Stability** (0-1): Consistency of execution time
```julia
stability = 1.0 / (1.0 + std(times) / 1e9)
```

**Memory** (0-1): Allocations normalized
```julia
memory = min(allocations / 1000.0, 1.0)
```

**Composite Score** (0-1000):
```julia
score = weighted_sum([
    accuracy * 0.3,
    speed * 0.2,
    (1-complexity) * 0.1,
    coherence * 0.15,
    stability * 0.15,
    (1-memory) * 0.1
]) * 1000
```

## Configuration

### Edit Metric Weights
Click **Config → Edit Weights** to customize:
- accuracy: 0.3 (default)
- speed: 0.2
- complexity: 0.1
- coherence: 0.15
- stability: 0.15
- memory: 0.1

### Convergence Settings
Edit `EvolutionConfig` in code:
```python
convergence_window: int = 5      # Check last N generations
convergence_threshold: float = 0.01  # Score variance threshold
min_score: float = 300.0         # Minimum to be considered
```

## Output Files

### algorithm_evolution_winners.md
Contains all winners categorized by quality:
- Generation number
- Round number
- All metrics
- Full source code
- Grand Champion section

## Tips for Best Results

### 1. Start Simple
Begin with a naive but correct implementation. The AI will optimize it.

### 2. Good Test Data
- Use representative inputs
- Test edge cases separately
- Ensure expected output is exact

### 3. Model Selection
- **4B models**: Fast, good for simple optimizations
- **7B models**: Better at complex algorithmic improvements
- **Coder-specific models**: Best understanding of optimization patterns

### 4. Convergence
- Let it run until "Converged" message
- Usually 5-15 generations for simple algorithms
- Complex algorithms may need more rounds

### 5. Review Winners
Bad algorithms won't win due to:
- Minimum score threshold (300/1000)
- Accuracy-first weighting (30%)
- Syntax validation before benchmarking

## Troubleshooting

### "Ollama not running"
```bash
ollama serve
```

### "Julia not found"
Add Julia to PATH or install from https://julialang.org/

### "BenchmarkTools not installed"
GUI will offer to auto-install, or run:
```bash
julia -e 'using Pkg; Pkg.add("BenchmarkTools")'
```

### "X11 BadLength error"
Already fixed - all emojis removed from display

### Evolution not improving
- Check initial code is correct
- Verify test data is accurate
- Try different model
- Adjust metric weights

## Advanced Usage

### Custom Metrics
Add custom metrics by editing `AlgorithmMetrics` class and benchmark code.

### Integration with Tournament System
Use generated winners as candidates in your Julia tournament files.

### Batch Evolution
Load multiple functions and evolve them sequentially.

## Architecture

```
User Input (Julia Code)
    ↓
Syntax Validation
    ↓
Initial Benchmarking (BenchmarkTools.jl)
    ↓
Weakness Analysis
    ↓
Ollama Code Generation (6 variants)
    ↓
Parallel Evaluation
    ↓
Best Selection (above min_score)
    ↓
Convergence Check
    ↓
Winner Tracking
```

## Performance Notes

- **Small models (4B-7B)** work well on 6GB CUDA
- Each generation takes ~30-60s depending on model
- Julia compilation overhead on first run
- Subsequent runs are faster (JIT compiled)

## License

Same as parent project (Sovwave)

## Contributing

Improvements welcome! Focus areas:
- Additional optimization patterns in prompts
- Better complexity estimation
- Multi-objective optimization
- Parallel variant generation
