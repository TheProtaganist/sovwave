#!/usr/bin/env python3
"""
Ollama Algorithm Evolution GUI
================================
Uses small LLMs (4B-7B) to iteratively evolve optimal Julia algorithms through competitive rounds.

Features:
- Multi-round evolution with tournament-style competition
- Comprehensive metrics: accuracy, speed, complexity, coherence, stability, memory
- Custom user-defined metrics support
- Automatic convergence detection (finds optimal peak)
- Winners tracking in markdown files
- Import-aware Julia code generation
- Prevents bad algorithms from winning (validation + scoring)
- Uses BenchmarkTools.jl for accurate Julia performance metrics

Path of Least Resistance Categories:
- PERFECT: Flawless, mathematically exact, optimal
- EXCELLENT: Great approximation, minimal tradeoffs
- COMPROMISE: Acceptable tradeoffs
- BROKEN: Discarded
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import json
import re
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
import threading
import queue
from datetime import datetime


@dataclass
class AlgorithmMetrics:
    """Comprehensive algorithm performance metrics"""
    name: str
    generation: int
    round_num: int
    
    # Core metrics (0.0 - 1.0 scale)
    accuracy: float = 0.0
    speed: float = 0.0  # Higher is better (inverted from time)
    complexity: float = 0.0  # Lower is better (inverted)
    coherence: float = 0.0
    stability: float = 0.0
    memory: float = 0.0  # Lower is better (inverted)
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Raw values
    time_ns: float = 0.0
    allocations: int = 0
    convergence_loss: float = 0.0
    
    # Composite score
    score: float = 0.0
    category: str = "UNKNOWN"  # PERFECT, EXCELLENT, COMPROMISE, BROKEN
    
    # Code
    code: str = ""
    imports: List[str] = field(default_factory=list)
    
    def compute_composite_score(self, weights: Dict[str, float]) -> float:
        """Compute weighted composite score"""
        base_metrics = {
            'accuracy': self.accuracy,
            'speed': self.speed,
            'complexity': 1.0 - self.complexity,  # Invert (lower is better)
            'coherence': self.coherence,
            'stability': self.stability,
            'memory': 1.0 - self.memory  # Invert (lower is better)
        }
        
        # Add custom metrics
        base_metrics.update(self.custom_metrics)
        
        # Weighted sum
        score = 0.0
        total_weight = 0.0
        for metric, value in base_metrics.items():
            if metric in weights:
                score += weights[metric] * value
                total_weight += weights[metric]
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        # Multiply by 1000 for readability
        return score * 1000.0
    
    def categorize(self, threshold_perfect: float = 950.0, 
                   threshold_excellent: float = 750.0,
                   threshold_compromise: float = 500.0) -> str:
        """Categorize algorithm quality"""
        if self.score >= threshold_perfect:
            return "PERFECT"
        elif self.score >= threshold_excellent:
            return "EXCELLENT"
        elif self.score >= threshold_compromise:
            return "COMPROMISE"
        else:
            return "BROKEN"


@dataclass
class EvolutionConfig:
    """Evolution configuration"""
    rounds: int = -1  # -1 for unlimited
    algorithms_per_round: int = 6  # 2-16 algorithms per round
    convergence_window: int = 5  # Check last N generations
    convergence_threshold: float = 0.01  # Score variance threshold
    
    # Variant strategy
    variant_ratio: float = 0.5  # 50% variants, 50% new algorithms
    enforce_code_difference: bool = True  # Prevent lazy parameter changes
    min_code_difference_ratio: float = 0.3  # At least 30% code must be different
    
    # Metric weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.3,
        'speed': 0.2,
        'complexity': 0.1,
        'coherence': 0.15,
        'stability': 0.15,
        'memory': 0.1
    })
    
    # Quality thresholds
    min_score: float = 300.0  # Minimum score to be considered
    
    # Ollama settings
    ollama_model: str = "qwen2.5-coder:7b"
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Auto-validation
    auto_calculate_expected: bool = True
    validation_runs: int = 5  # Number of runs to verify expected value


class OllamaClient:
    """Ollama API client for code generation"""
    
    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def check_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ['curl', '-s', f'{self.base_url}/api/tags'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                models = []
                
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        # Extract model name (first column)
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                
                return models
            return []
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []
    
    def generate_algorithm(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate algorithm code using Ollama"""
        try:
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "options": {
                    "num_predict": 2000,
                    "top_p": 0.9
                }
            }
            
            result = subprocess.run(
                ['curl', '-s', '-X', 'POST', 
                 f'{self.base_url}/api/generate',
                 '-d', json.dumps(request_data)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get('response', '')
            return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""
    
    def improve_algorithm(self, current_code: str, metrics: AlgorithmMetrics, 
                         focus_areas: List[str]) -> str:
        """Generate improved version (variant) of algorithm"""
        focus_str = ", ".join(focus_areas)
        
        prompt = f"""You are an expert Julia algorithm optimization engineer.

Current algorithm performance:
- Accuracy: {metrics.accuracy:.2%}
- Speed: {metrics.speed:.2%}
- Complexity: {metrics.complexity:.2%}
- Overall Score: {metrics.score:.2f}/1000

Current Julia code:
```julia
{current_code}
```

Task: Improve this Julia algorithm focusing on: {focus_str}

CRITICAL REQUIREMENTS:
1. Keep the SAME function signature and name
2. Improve performance in the focus areas
3. Maintain or improve accuracy (NEVER decrease accuracy)
4. Use FUNDAMENTALLY DIFFERENT logic/algorithm (not just parameter tweaks)
5. Consider: different data structures, algorithms, loop orders, mathematical identities
6. Use Julia optimizations: type annotations, @inbounds, @simd, vectorization
7. Return ONLY the improved Julia function code, NO explanations

Improved Julia code:"""
        
        return self.generate_algorithm(prompt, temperature=0.5)
    
    def generate_new_algorithm(self, func_name: str, func_signature: str,
                               description: str, test_input: str, expected_output: str) -> str:
        """Generate completely new algorithm (not based on existing code)"""
        
        prompt = f"""You are an expert Julia algorithm engineer.

Task: Create a NEW, ORIGINAL Julia function that solves this problem.

Function Requirements:
- Name: {func_name}
- Signature: {func_signature}
- Purpose: {description}
- Must return: {expected_output} for input: {test_input}

CRITICAL REQUIREMENTS:
1. Create a COMPLETELY DIFFERENT algorithmic approach
2. DO NOT copy patterns from previous attempts
3. Use creative data structures or mathematical approaches
4. Optimize for speed and accuracy
5. Use Julia-specific optimizations
6. Include type annotations
7. Return ONLY the Julia function code, NO explanations

Examples of different approaches:
- Iterative vs recursive vs matrix math vs lookup table
- Different loop structures, vectorization, or mathematical identities
- Memoization, dynamic programming, divide-and-conquer
- Bit manipulation, algebraic shortcuts, precomputation

New Julia code:"""
        
        return self.generate_algorithm(prompt, temperature=0.8)


class CodeExecutor:
    """Execute and benchmark Julia code safely"""
    
    def __init__(self):
        self.test_data = None
        self.expected_output = None
        self.julia_cmd = "julia"
        self.project_packages = []  # Track external packages
    
    def calculate_expected_value(self, code: str, func_name: str, test_input: Any, 
                                 validation_runs: int = 5) -> Tuple[bool, Any, str]:
        """Auto-calculate expected value by running reference implementation"""
        try:
            julia_input = self.python_to_julia(test_input)
            
            test_code = f'''
{code}

# Run function multiple times to verify consistency
results = []
for _ in 1:{validation_runs}
    try
        result = {func_name}({julia_input})
        push!(results, result)
    catch e
        println("ERROR: ", e)
        exit(1)
    end
end

# Check all results are the same
if length(results) == 0
    println("ERROR: No results generated")
    exit(1)
end

first_result = results[1]
all_same = all(r == first_result for r in results)

if !all_same
    println("INCONSISTENT: Results vary across runs")
    exit(1)
end

println("EXPECTED:", first_result)
'''
            
            result = subprocess.run(
                [self.julia_cmd, '-e', test_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False, None, result.stderr
            
            # Extract expected value
            expected_match = re.search(r'EXPECTED:(.+)', result.stdout)
            if expected_match:
                expected_str = expected_match.group(1).strip()
                # Parse Julia output to Python value
                expected = self.julia_to_python(expected_str)
                return True, expected, ""
            
            return False, None, "Could not extract expected value"
            
        except Exception as e:
            return False, None, str(e)
    
    def julia_to_python(self, julia_str: str) -> Any:
        """Convert Julia output string to Python value"""
        julia_str = julia_str.strip()
        
        # Boolean
        if julia_str == "true":
            return True
        if julia_str == "false":
            return False
        
        # Try numeric
        try:
            if '.' in julia_str:
                return float(julia_str)
            else:
                return int(julia_str)
        except:
            pass
        
        # String
        return julia_str
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract using/import statements from code"""
        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('using ') or line.startswith('import '):
                imports.append(line)
        return imports
    
    def calculate_code_difference(self, code1: str, code2: str) -> float:
        """Calculate how different two code snippets are (0.0 = identical, 1.0 = completely different)"""
        # Normalize: remove comments, whitespace, and formatting
        def normalize(code):
            lines = []
            for line in code.split('\n'):
                # Remove comments
                line = re.sub(r'#.*$', '', line)
                # Remove docstrings
                line = re.sub(r'""".*?"""', '', line, flags=re.DOTALL)
                # Strip whitespace
                line = line.strip()
                if line:
                    lines.append(line)
            return ' '.join(lines)
        
        norm1 = normalize(code1)
        norm2 = normalize(code2)
        
        if not norm1 or not norm2:
            return 1.0
        
        # Calculate Levenshtein-like similarity
        len1, len2 = len(norm1), len(norm2)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 0.0
        
        # Simple character-level difference
        common_chars = sum(1 for c1, c2 in zip(norm1, norm2) if c1 == c2)
        difference = 1.0 - (common_chars / max_len)
        
        # Also check for structural differences (keywords, patterns)
        keywords1 = set(re.findall(r'\b(for|while|if|else|function|return|end)\b', code1))
        keywords2 = set(re.findall(r'\b(for|while|if|else|function|return|end)\b', code2))
        
        if keywords1 or keywords2:
            keyword_diff = len(keywords1.symmetric_difference(keywords2)) / max(len(keywords1 | keywords2), 1)
            # Weight both character and structural differences
            difference = (difference * 0.7) + (keyword_diff * 0.3)
        
        return min(difference, 1.0)
    
    def is_sufficiently_different(self, new_code: str, reference_code: str, 
                                  min_difference: float = 0.3) -> bool:
        """Check if new code is sufficiently different from reference"""
        diff = self.calculate_code_difference(new_code, reference_code)
        return diff >= min_difference
    
    def extract_function(self, code: str) -> Optional[str]:
        """Extract function definition from code"""
        # Remove markdown code blocks
        code = re.sub(r'```julia\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        
        # Find function definition
        func_match = re.search(r'function\s+(\w+!?)\s*\([^)]*\)', code)
        if func_match:
            return func_match.group(1)
        
        # Short form: name(...) = ...
        short_match = re.search(r'^(\w+!?)\s*\([^)]*\)\s*=', code, re.MULTILINE)
        if short_match:
            return short_match.group(1)
        
        return None
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate Julia syntax"""
        try:
            # Use Julia's parser to check syntax
            test_code = f'''
try
    Meta.parse("""
{code}
""")
    println("SYNTAX_OK")
catch e
    println("SYNTAX_ERROR: ", e)
end
'''
            result = subprocess.run(
                [self.julia_cmd, '-e', test_code],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'SYNTAX_OK' in result.stdout:
                return True, ""
            else:
                return False, result.stdout
        except Exception as e:
            return False, str(e)
    
    def benchmark_function(self, code: str, func_name: str, 
                          test_input: Any, expected: Any) -> AlgorithmMetrics:
        """Execute and benchmark Julia function"""
        metrics = AlgorithmMetrics(name=func_name, generation=0, round_num=0)
        
        # Validate syntax
        valid, error = self.validate_code(code)
        if not valid:
            metrics.category = "BROKEN"
            return metrics
        
        try:
            # Create benchmark script
            # Convert Python values to Julia syntax
            julia_input = self.python_to_julia(test_input)
            julia_expected = self.python_to_julia(expected)
            
            benchmark_code = f'''
using BenchmarkTools

{code}

# Test correctness
function test_correctness()
    correct = 0
    total = 100
    
    for _ in 1:total
        try
            result = {func_name}({julia_input})
            expected = {julia_expected}
            
            # Check if result matches expected
            if isapprox(result, expected, atol=1e-6) || result == expected
                correct += 1
            end
        catch
            # Failed execution
        end
    end
    
    return correct / total
end

# Benchmark speed
function benchmark_speed()
    try
        bench = @benchmark {func_name}({julia_input})
        return mean(bench.times), minimum(bench.times), std(bench.times), bench.allocs
    catch
        return Inf, Inf, Inf, 0
    end
end

accuracy = test_correctness()
mean_time, min_time, std_time, allocs = benchmark_speed()

println("ACCURACY:", accuracy)
println("TIME_NS:", mean_time)
println("MIN_TIME_NS:", min_time)
println("STD_TIME_NS:", std_time)
println("ALLOCS:", allocs)
'''
            
            # Run Julia benchmark
            result = subprocess.run(
                [self.julia_cmd, '-e', benchmark_code],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                metrics.category = "BROKEN"
                return metrics
            
            # Parse results
            output = result.stdout
            
            # Extract metrics from output
            accuracy_match = re.search(r'ACCURACY:([\d.]+)', output)
            time_match = re.search(r'TIME_NS:([\d.]+)', output)
            std_match = re.search(r'STD_TIME_NS:([\d.]+)', output)
            allocs_match = re.search(r'ALLOCS:(\d+)', output)
            
            if accuracy_match:
                metrics.accuracy = float(accuracy_match.group(1))
            
            if time_match:
                metrics.time_ns = float(time_match.group(1))
                # Normalize speed (higher is better)
                metrics.speed = 1.0 / (1.0 + metrics.time_ns / 1e6)
            
            if allocs_match:
                metrics.allocations = int(allocs_match.group(1))
                # Memory score (lower allocations = better)
                metrics.memory = min(metrics.allocations / 1000.0, 1.0)
            
            # Stability (from time variance)
            if std_match:
                std_time = float(std_match.group(1))
                metrics.stability = 1.0 / (1.0 + std_time / 1e9)
            
            # Complexity (estimate from code)
            metrics.complexity = self.estimate_complexity(code)
            
            # Coherence (code quality)
            metrics.coherence = self.estimate_coherence(code)
            
            metrics.code = code
            
        except subprocess.TimeoutExpired:
            metrics.category = "BROKEN"
            print("Execution timed out")
        except Exception as e:
            metrics.category = "BROKEN"
            print(f"Execution error: {e}")
        
        return metrics
    
    def python_to_julia(self, value: Any) -> str:
        """Convert Python value to Julia syntax"""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, list):
            items = ", ".join(self.python_to_julia(v) for v in value)
            return f'[{items}]'
        elif isinstance(value, tuple):
            items = ", ".join(self.python_to_julia(v) for v in value)
            return f'({items})'
        else:
            return str(value)
    
    def estimate_complexity(self, code: str) -> float:
        """Estimate algorithmic complexity from Julia code"""
        # Count nested loops
        lines = code.split('\n')
        loop_depth = 0
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Count loop starts
            if re.match(r'(for\s+|while\s+)', stripped):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            # Count ends
            if stripped == 'end':
                current_depth = max(0, current_depth - 1)
        
        # Estimate complexity score (0.0 = O(1), 1.0 = O(n^3+))
        complexity_map = {0: 0.1, 1: 0.3, 2: 0.6, 3: 0.9}
        return complexity_map.get(max_depth, 1.0)
    
    def estimate_coherence(self, code: str) -> float:
        """Estimate Julia code coherence/quality"""
        score = 1.0
        lines = code.split('\n')
        
        # Penalize overly long lines
        avg_line_length = sum(len(l) for l in lines) / max(len(lines), 1)
        if avg_line_length > 100:
            score *= 0.8
        
        # Reward docstrings
        if '"""' in code:
            score *= 1.1
        
        # Reward type annotations
        if '::' in code:
            score *= 1.05
        
        # Reward @inbounds, @simd optimization macros
        if '@inbounds' in code or '@simd' in code:
            score *= 1.1
        
        # Penalize excessive nesting
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 16:
            score *= 0.9
        
        return min(score, 1.0)


class AlgorithmEvolutionEngine:
    """Core evolution engine"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.ollama = OllamaClient(config.ollama_model)
        self.executor = CodeExecutor()
        
        self.generations: List[List[AlgorithmMetrics]] = []
        self.best_ever: Optional[AlgorithmMetrics] = None
        self.convergence_count = 0
    
    def evolve(self, initial_code: str, func_name: str, 
               test_input: Any, expected: Any,
               callback: Optional[Callable] = None) -> List[AlgorithmMetrics]:
        """Run evolution process with multiple algorithms per round"""
        
        generation = 0
        round_num = 0
        
        # Auto-calculate expected value if enabled
        if self.config.auto_calculate_expected and expected is None:
            success, calc_expected, error = self.executor.calculate_expected_value(
                initial_code, func_name, test_input, self.config.validation_runs
            )
            if success:
                expected = calc_expected
                print(f"Auto-calculated expected value: {expected}")
                if callback:
                    callback(f"[AUTO] Expected value: {expected}")
            else:
                print(f"Failed to auto-calculate expected: {error}")
                if callback:
                    callback(f"[ERROR] Could not calculate expected value: {error}")
                return []
        
        # Extract imports from initial code
        self.executor.project_packages = self.executor.extract_imports(initial_code)
        
        # Evaluate initial algorithm
        current_best = self.executor.benchmark_function(initial_code, func_name, test_input, expected)
        current_best.generation = generation
        current_best.round_num = round_num
        current_best.name = f"{func_name}_Gen{generation}_Base"
        current_best.score = current_best.compute_composite_score(self.config.weights)
        current_best.category = current_best.categorize()
        
        self.generations.append([current_best])
        self.best_ever = current_best
        
        if callback:
            callback(current_best)
        
        # Track all generated codes to ensure diversity
        all_codes = [initial_code]
        
        # Evolution loop
        while True:
            generation += 1
            
            # Check round limit
            if self.config.rounds > 0 and round_num >= self.config.rounds:
                break
            
            # Calculate how many variants vs new algorithms
            n_algorithms = self.config.algorithms_per_round
            n_variants = int(n_algorithms * self.config.variant_ratio)
            n_new = n_algorithms - n_variants
            
            print(f"\nGeneration {generation}: {n_variants} variants + {n_new} new algorithms")
            
            # Generate algorithms
            algorithms = []
            
            # Focus areas based on weakest metrics
            weak_metrics = self.identify_weak_metrics(current_best)
            
            # 1. Generate VARIANTS (improved versions of current best)
            variant_attempts = 0
            max_variant_attempts = n_variants * 3  # Allow retries
            
            while len(algorithms) < n_variants and variant_attempts < max_variant_attempts:
                variant_attempts += 1
                
                improved_code = self.ollama.improve_algorithm(
                    current_best.code,
                    current_best,
                    weak_metrics
                )
                
                if improved_code:
                    # Check if sufficiently different
                    if self.config.enforce_code_difference:
                        is_different = all(
                            self.executor.is_sufficiently_different(
                                improved_code, existing_code, 
                                self.config.min_code_difference_ratio
                            )
                            for existing_code in all_codes
                        )
                        
                        if not is_different:
                            print(f"  [REJECTED] Variant too similar to existing code")
                            continue
                    
                    metrics = self.executor.benchmark_function(
                        improved_code, func_name, test_input, expected
                    )
                    metrics.generation = generation
                    metrics.round_num = round_num
                    metrics.name = f"{func_name}_Gen{generation}_Variant{len(algorithms)+1}"
                    metrics.score = metrics.compute_composite_score(self.config.weights)
                    metrics.category = metrics.categorize()
                    
                    # Only keep if better than minimum threshold
                    if metrics.score >= self.config.min_score:
                        algorithms.append(metrics)
                        all_codes.append(improved_code)
                        
                        if callback:
                            callback(metrics)
            
            # 2. Generate NEW ALGORITHMS (completely different approaches)
            # Extract function signature from initial code
            func_signature_match = re.search(
                rf'function\s+{func_name}\s*\(([^)]*)\)(?:::\s*(\w+))?',
                initial_code
            )
            
            func_signature = f"{func_name}(input)"
            if func_signature_match:
                params = func_signature_match.group(1)
                return_type = func_signature_match.group(2) or "Any"
                func_signature = f"{func_name}({params})::{return_type}"
            
            # Description from docstring or generic
            description = "Solve the algorithmic problem"
            docstring_match = re.search(r'"""([^"]+)"""', initial_code)
            if docstring_match:
                description = docstring_match.group(1).strip()
            
            new_attempts = 0
            max_new_attempts = n_new * 3
            
            while len(algorithms) < n_algorithms and new_attempts < max_new_attempts:
                new_attempts += 1
                
                new_code = self.ollama.generate_new_algorithm(
                    func_name,
                    func_signature,
                    description,
                    str(test_input),
                    str(expected)
                )
                
                if new_code:
                    # Check if sufficiently different from all existing codes
                    is_different = all(
                        self.executor.is_sufficiently_different(
                            new_code, existing_code,
                            self.config.min_code_difference_ratio
                        )
                        for existing_code in all_codes
                    )
                    
                    if not is_different:
                        print(f"  [REJECTED] New algorithm too similar to existing code")
                        continue
                    
                    metrics = self.executor.benchmark_function(
                        new_code, func_name, test_input, expected
                    )
                    metrics.generation = generation
                    metrics.round_num = round_num
                    metrics.name = f"{func_name}_Gen{generation}_New{len(algorithms)-n_variants+1}"
                    metrics.score = metrics.compute_composite_score(self.config.weights)
                    metrics.category = metrics.categorize()
                    
                    if metrics.score >= self.config.min_score:
                        algorithms.append(metrics)
                        all_codes.append(new_code)
                        
                        if callback:
                            callback(metrics)
            
            if not algorithms:
                print(f"Generation {generation}: No valid algorithms generated")
                break
            
            # Find best from this generation
            best_this_gen = max(algorithms, key=lambda m: m.score)
            
            self.generations.append(algorithms)
            
            # Check if improved
            if best_this_gen.score > current_best.score:
                print(f"  [IMPROVEMENT] {best_this_gen.name}: {best_this_gen.score:.2f}")
                current_best = best_this_gen
                self.convergence_count = 0
                
                # Update best ever
                if best_this_gen.score > self.best_ever.score:
                    self.best_ever = best_this_gen
            else:
                self.convergence_count += 1
            
            # Check convergence
            if self.has_converged():
                print(f"Converged after {generation} generations!")
                break
            
            round_num += 1
        
        return self.generations[-1] if self.generations else []
    
    def identify_weak_metrics(self, metrics: AlgorithmMetrics) -> List[str]:
        """Identify weakest metrics to focus improvement"""
        metric_values = {
            'accuracy': metrics.accuracy,
            'speed': metrics.speed,
            'complexity': 1.0 - metrics.complexity,
            'coherence': metrics.coherence,
            'stability': metrics.stability,
            'memory': 1.0 - metrics.memory
        }
        
        # Sort by value (lowest first)
        sorted_metrics = sorted(metric_values.items(), key=lambda x: x[1])
        
        # Return bottom 3
        return [m[0] for m in sorted_metrics[:3]]
    
    def has_converged(self) -> bool:
        """Check if evolution has converged"""
        if self.convergence_count >= self.config.convergence_window:
            return True
        
        # Check score variance in recent generations
        if len(self.generations) >= self.config.convergence_window:
            recent_scores = []
            for gen in self.generations[-self.config.convergence_window:]:
                if gen:
                    recent_scores.append(max(m.score for m in gen))
            
            if recent_scores:
                variance = sum((s - sum(recent_scores)/len(recent_scores))**2 
                              for s in recent_scores) / len(recent_scores)
                
                if variance < self.config.convergence_threshold:
                    return True
        
        return False


class WinnersTracker:
    """Track and persist winners to markdown"""
    
    def __init__(self, filepath: str = "algorithm_evolution_winners.md"):
        self.filepath = filepath
        self.winners: Dict[str, List[AlgorithmMetrics]] = {
            'PERFECT': [],
            'EXCELLENT': [],
            'COMPROMISE': [],
            'BROKEN': []
        }
        self.load()
    
    def load(self):
        """Load existing winners from file"""
        if Path(self.filepath).exists():
            try:
                with open(self.filepath, 'r') as f:
                    content = f.read()
                    # Parse markdown (simple parsing)
                    # This is simplified - full implementation would parse the md structure
                    pass
            except Exception as e:
                print(f"Failed to load winners: {e}")
    
    def add_winner(self, metrics: AlgorithmMetrics):
        """Add new winner"""
        self.winners[metrics.category].append(metrics)
    
    def save(self):
        """Save winners to markdown"""
        try:
            with open(self.filepath, 'w') as f:
                f.write("# Algorithm Evolution Winners\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for category in ['PERFECT', 'EXCELLENT', 'COMPROMISE']:
                    winners = sorted(self.winners[category], 
                                   key=lambda m: m.score, reverse=True)
                    
                    icon = {'PERFECT': '[PERFECT]', 'EXCELLENT': '[EXCELLENT]', 'COMPROMISE': '[COMPROMISE]'}[category]
                    
                    f.write(f"## {icon} {category} Algorithms ({len(winners)})\n\n")
                    
                    for w in winners[:20]:  # Top 20
                        f.write(f"- **{w.name}** - Score: {w.score:.2f}\n")
                        f.write(f"  - Accuracy: {w.accuracy:.2%}, Speed: {w.speed:.2%}, ")
                        f.write(f"Complexity: {w.complexity:.2%}\n")
                        f.write(f"  - Generation: {w.generation}, Round: {w.round_num}\n\n")
                
                # Add grand champion
                if self.winners['PERFECT'] or self.winners['EXCELLENT']:
                    all_top = self.winners['PERFECT'] + self.winners['EXCELLENT']
                    champion = max(all_top, key=lambda m: m.score)
                    
                    f.write("\n## [WINNER] Grand Champion\n\n")
                    f.write(f"**{champion.name}**\n\n")
                    f.write(f"- Score: {champion.score:.2f}\n")
                    f.write(f"- Accuracy: {champion.accuracy:.2%}\n")
                    f.write(f"- Speed: {champion.speed:.2%}\n")
                    f.write(f"- Complexity: {champion.complexity:.2%}\n")
                    f.write(f"- Coherence: {champion.coherence:.2%}\n")
                    f.write(f"- Stability: {champion.stability:.2%}\n")
                    f.write(f"- Memory: {champion.memory:.2%}\n")
                    f.write(f"- Generation: {champion.generation}\n")
                    f.write(f"- Round: {champion.round_num}\n\n")
                    f.write("```python\n")
                    f.write(champion.code)
                    f.write("\n```\n")
        
        except Exception as e:
            print(f"Failed to save winners: {e}")


class EvolutionGUI:
    """Main GUI for algorithm evolution"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Algorithm Evolution Engine")
        self.root.geometry("1400x900")
        
        self.config = EvolutionConfig()
        self.engine: Optional[AlgorithmEvolutionEngine] = None
        self.tracker = WinnersTracker()
        
        self.current_code = ""
        self.func_name = ""
        self.is_running = False
        self.available_models = []
        
        self.setup_ui()
        self.check_ollama()
    
    def setup_ui(self):
        """Setup main UI"""
        # Menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Code", command=self.load_code)
        file_menu.add_command(label="Load Winners", command=self.load_winners)
        file_menu.add_command(label="Save Winners", command=self.save_winners)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Config", menu=config_menu)
        config_menu.add_command(label="Edit Weights", command=self.edit_weights)
        config_menu.add_command(label="Edit Settings", command=self.edit_settings)
        
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="[DNA] Algorithm Evolution Engine", 
                        font=('TkDefaultFont', 14, 'bold'))
        title.pack(pady=(0, 10))
        
        # Top controls
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(control_frame, text="Function (auto if empty):").pack(side=tk.LEFT)
        self.func_entry = tk.Entry(control_frame, width=20)
        self.func_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Algorithms/Round:").pack(side=tk.LEFT, padx=(20, 0))
        self.algos_entry = tk.Entry(control_frame, width=5)
        self.algos_entry.insert(0, "6")
        self.algos_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Rounds:").pack(side=tk.LEFT, padx=(10, 0))
        self.rounds_entry = tk.Entry(control_frame, width=5)
        self.rounds_entry.insert(0, "-1")
        self.rounds_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(20, 0))
        self.model_var = tk.StringVar(value="qwen2.5-coder:7b")
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                   values=["qwen2.5-coder:7b", "qwen2.5-coder:4b", 
                                          "codellama:7b", "deepseek-coder:6.7b"],
                                   width=25)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Start Evolution", command=self.start_evolution,
                 bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Stop", command=self.stop_evolution,
                 bg='#f44336', fg='white').pack(side=tk.LEFT)
        
        # Test data frame
        test_frame = tk.LabelFrame(main_frame, text="Test Data (Optional - Auto-calculates if empty)")
        test_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(test_frame, text="Input:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.test_input_entry = tk.Entry(test_frame, width=40)
        self.test_input_entry.insert(0, "10")
        self.test_input_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(test_frame, text="Expected (leave empty to auto-calculate):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.test_expected_entry = tk.Entry(test_frame, width=40)
        self.test_expected_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Options frame
        options_frame = tk.LabelFrame(main_frame, text="Options")
        options_frame.pack(fill=tk.X, pady=5)
        
        self.auto_calc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Auto-calculate expected values", 
                      variable=self.auto_calc_var).pack(side=tk.LEFT, padx=10)
        
        self.enforce_diff_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Enforce code differences (no lazy parameter changes)",
                      variable=self.enforce_diff_var).pack(side=tk.LEFT, padx=10)
        
        # Split panes
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Code editor
        left_frame = tk.Frame(paned)
        paned.add(left_frame, width=500)
        
        tk.Label(left_frame, text="Initial Julia Code", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
        self.code_text = scrolledtext.ScrolledText(left_frame, wrap=tk.NONE, height=30)
        self.code_text.pack(fill=tk.BOTH, expand=True)
        self.code_text.insert('1.0', '''function fibonacci(n::Int)::Int
    """Calculate nth Fibonacci number using dynamic programming"""
    if n <= 1
        return n
    end
    
    a, b = 0, 1
    for i in 2:n
        a, b = b, a + b
    end
    return b
end
''')
        
        # Right: Results tabs
        right_frame = tk.Frame(paned)
        paned.add(right_frame)
        
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Evolution tab
        evolution_frame = tk.Frame(notebook)
        notebook.add(evolution_frame, text="Evolution")
        self.evolution_text = scrolledtext.ScrolledText(evolution_frame, wrap=tk.WORD)
        self.evolution_text.pack(fill=tk.BOTH, expand=True)
        
        # Winners tab
        winners_frame = tk.Frame(notebook)
        notebook.add(winners_frame, text="Winners")
        self.winners_text = scrolledtext.ScrolledText(winners_frame, wrap=tk.WORD)
        self.winners_text.pack(fill=tk.BOTH, expand=True)
        
        # Best Code tab
        best_frame = tk.Frame(notebook)
        notebook.add(best_frame, text="Best Code")
        self.best_code_text = scrolledtext.ScrolledText(best_frame, wrap=tk.NONE)
        self.best_code_text.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        metrics_frame = tk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        for text_widget in [self.evolution_text, self.winners_text, self.metrics_text]:
            text_widget.tag_config('perfect', foreground='green', font=('TkDefaultFont', 10, 'bold'))
            text_widget.tag_config('excellent', foreground='blue')
            text_widget.tag_config('compromise', foreground='orange')
            text_widget.tag_config('broken', foreground='red')
            text_widget.tag_config('header', font=('TkDefaultFont', 11, 'bold'))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var,
                            relief=tk.SUNKEN, anchor=tk.W, bg='lightgray')
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def check_ollama(self):
        """Check if Ollama is available"""
        client = OllamaClient()
        if client.check_available():
            # Get available models
            self.available_models = client.list_models()
            
            if self.available_models:
                # Update combobox with available models
                self.model_combo['values'] = self.available_models
                self.model_var.set(self.available_models[0])
                self.status_var.set(f"Ollama: Connected ({len(self.available_models)} models)")
            else:
                self.status_var.set("Ollama: Connected (no models found)")
        else:
            self.status_var.set("WARNING: Ollama not running")
            messagebox.showwarning("Ollama Not Found",
                                  "Ollama is not running. Please start Ollama service:\n"
                                  "ollama serve")
        
        # Check Julia and BenchmarkTools
        self.check_julia()
    
    def check_julia(self):
        """Check if Julia and BenchmarkTools are available"""
        try:
            # Check Julia
            result = subprocess.run(
                ['julia', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                messagebox.showwarning("Julia Not Found",
                                      "Julia is not installed or not in PATH.\n"
                                      "Please install Julia: https://julialang.org/downloads/")
                return
            
            # Check BenchmarkTools
            result = subprocess.run(
                ['julia', '-e', 'using BenchmarkTools; println("OK")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "OK" not in result.stdout:
                response = messagebox.askyesno("BenchmarkTools Not Found",
                                               "BenchmarkTools.jl is required but not installed.\n\n"
                                               "Install it now? (This will run: using Pkg; Pkg.add(\"BenchmarkTools\"))")
                if response:
                    self.install_benchmark_tools()
        
        except Exception as e:
            messagebox.showwarning("Julia Check Failed", f"Could not verify Julia installation:\n{e}")
    
    def install_benchmark_tools(self):
        """Install BenchmarkTools.jl"""
        try:
            self.status_var.set("Installing BenchmarkTools.jl...")
            self.root.update()
            
            result = subprocess.run(
                ['julia', '-e', 'using Pkg; Pkg.add("BenchmarkTools")'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "BenchmarkTools.jl installed successfully!")
                self.status_var.set("Ready")
            else:
                messagebox.showerror("Installation Failed", 
                                    f"Failed to install BenchmarkTools.jl:\n{result.stderr}")
                self.status_var.set("BenchmarkTools.jl installation failed")
        
        except subprocess.TimeoutExpired:
            messagebox.showerror("Timeout", "Installation timed out. Please install manually:\n"
                                           "julia -e 'using Pkg; Pkg.add(\"BenchmarkTools\")'")
        except Exception as e:
            messagebox.showerror("Error", f"Installation error:\n{e}")
    
    def load_code(self):
        """Load code from file"""
        filename = filedialog.askopenfilename(
            title="Select Julia File",
            filetypes=[("Julia files", "*.jl"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r') as f:
                self.code_text.delete('1.0', tk.END)
                self.code_text.insert('1.0', f.read())
    
    def load_winners(self):
        """Load and display winners"""
        self.tracker.load()
        self.display_winners()
    
    def save_winners(self):
        """Save winners to file"""
        self.tracker.save()
        messagebox.showinfo("Success", f"Winners saved to {self.tracker.filepath}")
    
    def display_winners(self):
        """Display winners in UI"""
        self.winners_text.delete('1.0', tk.END)
        
        for category in ['PERFECT', 'EXCELLENT', 'COMPROMISE']:
            winners = sorted(self.tracker.winners[category],
                           key=lambda m: m.score, reverse=True)
            
            tag = category.lower()
            self.winners_text.insert(tk.END, f"\n{category} ({len(winners)})\n", 'header')
            self.winners_text.insert(tk.END, "=" * 50 + "\n\n")
            
            for w in winners[:10]:
                self.winners_text.insert(tk.END, f"{w.name} - Score: {w.score:.2f}\n", tag)
                self.winners_text.insert(tk.END, 
                    f"  Acc: {w.accuracy:.1%}, Spd: {w.speed:.1%}, Cplx: {w.complexity:.1%}\n")
    
    def edit_weights(self):
        """Open weight editor dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Metric Weights")
        dialog.geometry("400x300")
        
        tk.Label(dialog, text="Metric Weights", font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        entries = {}
        for metric, weight in self.config.weights.items():
            frame = tk.Frame(dialog)
            frame.pack(fill=tk.X, padx=20, pady=5)
            tk.Label(frame, text=f"{metric}:", width=15).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=10)
            entry.insert(0, str(weight))
            entry.pack(side=tk.LEFT, padx=5)
            entries[metric] = entry
        
        def save():
            for metric, entry in entries.items():
                try:
                    self.config.weights[metric] = float(entry.get())
                except:
                    pass
            dialog.destroy()
        
        tk.Button(dialog, text="Save", command=save, bg='#4CAF50', fg='white').pack(pady=10)
    
    def edit_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "Settings dialog not yet implemented")
    
    def start_evolution(self):
        """Start evolution process"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Evolution is already running")
            return
        
        # Get code first
        code = self.code_text.get('1.0', tk.END).strip()
        if not code:
            messagebox.showerror("Error", "Please provide initial code")
            return
        
        # Get or infer function name
        self.func_name = self.func_entry.get().strip()
        if not self.func_name:
            # Auto-infer from code
            executor = CodeExecutor()
            inferred_name = executor.extract_function(code)
            if inferred_name:
                self.func_name = inferred_name
                self.func_entry.delete(0, tk.END)
                self.func_entry.insert(0, inferred_name)
                print(f"Auto-inferred function name: {inferred_name}")
            else:
                messagebox.showerror("Error", "Could not infer function name. Please enter manually.")
                return
        
        try:
            rounds = int(self.rounds_entry.get())
            self.config.rounds = rounds
        except:
            self.config.rounds = -1
        
        try:
            n_algos = int(self.algos_entry.get())
            if 2 <= n_algos <= 16:
                self.config.algorithms_per_round = n_algos
            else:
                messagebox.showerror("Error", "Algorithms per round must be between 2 and 16")
                return
        except:
            messagebox.showerror("Error", "Invalid number of algorithms")
            return
        
        self.config.ollama_model = self.model_var.get()
        self.config.auto_calculate_expected = self.auto_calc_var.get()
        self.config.enforce_code_difference = self.enforce_diff_var.get()
        
        # Get test data
        try:
            test_input = eval(self.test_input_entry.get())
        except:
            messagebox.showerror("Error", "Invalid test input")
            return
        
        # Expected can be None for auto-calculation
        expected = None
        expected_str = self.test_expected_entry.get().strip()
        if expected_str:
            try:
                expected = eval(expected_str)
            except:
                messagebox.showerror("Error", "Invalid expected value")
                return
        
        # Initialize engine
        self.engine = AlgorithmEvolutionEngine(self.config)
        self.engine.executor.test_data = test_input
        self.engine.executor.expected_output = expected
        
        # Run in thread
        self.is_running = True
        self.evolution_text.delete('1.0', tk.END)
        
        def evolution_callback(data):
            if isinstance(data, AlgorithmMetrics):
                self.root.after(0, self.update_evolution_display, data)
            elif isinstance(data, str):
                # Status message
                self.root.after(0, lambda msg=data: self.evolution_text.insert(tk.END, msg + "\n"))
        
        def run():
            try:
                self.engine.evolve(code, self.func_name, test_input, expected, evolution_callback)
                self.root.after(0, self.evolution_complete)
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                self.is_running = False
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
        self.status_var.set(f"Evolution running... ({n_algos} algorithms/round)")
    
    def stop_evolution(self):
        """Stop evolution"""
        self.is_running = False
        self.status_var.set("Stopped")
    
    def update_evolution_display(self, metrics: AlgorithmMetrics):
        """Update evolution display with new metrics"""
        tag = metrics.category.lower()
        
        self.evolution_text.insert(tk.END, 
            f"[Gen {metrics.generation}] {metrics.name} - Score: {metrics.score:.2f} ", tag)
        self.evolution_text.insert(tk.END, f"({metrics.category})\n")
        self.evolution_text.insert(tk.END,
            f"  Acc: {metrics.accuracy:.1%}, Spd: {metrics.speed:.1%}, " 
            f"Cplx: {metrics.complexity:.1%}, Coh: {metrics.coherence:.1%}\n\n")
        self.evolution_text.see(tk.END)
        
        # Update metrics display
        self.update_metrics_display(metrics)
    
    def update_metrics_display(self, metrics: AlgorithmMetrics):
        """Update detailed metrics display"""
        self.metrics_text.delete('1.0', tk.END)
        
        self.metrics_text.insert(tk.END, f"Algorithm: {metrics.name}\n", 'header')
        self.metrics_text.insert(tk.END, f"Generation: {metrics.generation}\n")
        self.metrics_text.insert(tk.END, f"Category: {metrics.category}\n\n")
        
        self.metrics_text.insert(tk.END, "Core Metrics:\n", 'header')
        self.metrics_text.insert(tk.END, f"  Accuracy:    {metrics.accuracy:.2%}\n")
        self.metrics_text.insert(tk.END, f"  Speed:       {metrics.speed:.2%}\n")
        self.metrics_text.insert(tk.END, f"  Complexity:  {metrics.complexity:.2%}\n")
        self.metrics_text.insert(tk.END, f"  Coherence:   {metrics.coherence:.2%}\n")
        self.metrics_text.insert(tk.END, f"  Stability:   {metrics.stability:.2%}\n")
        self.metrics_text.insert(tk.END, f"  Memory:      {metrics.memory:.2%}\n\n")
        
        self.metrics_text.insert(tk.END, f"Composite Score: {metrics.score:.2f}/1000\n\n", 'header')
        
        self.metrics_text.insert(tk.END, "Raw Values:\n", 'header')
        self.metrics_text.insert(tk.END, f"  Time: {metrics.time_ns/1e6:.2f} ms\n")
        self.metrics_text.insert(tk.END, f"  Allocations: {metrics.allocations}\n")
    
    def evolution_complete(self):
        """Handle evolution completion"""
        if self.engine and self.engine.best_ever:
            best = self.engine.best_ever
            
            # Add to winners
            self.tracker.add_winner(best)
            self.tracker.save()
            
            # Display best code
            self.best_code_text.delete('1.0', tk.END)
            self.best_code_text.insert('1.0', best.code)
            
            # Show summary
            self.display_winners()
            
            self.status_var.set(f"Complete! Best: {best.score:.2f} ({best.category})")
            
            messagebox.showinfo("Evolution Complete",
                              f"Best Algorithm: {best.name}\n"
                              f"Score: {best.score:.2f}\n"
                              f"Category: {best.category}\n"
                              f"Generations: {best.generation}")
        else:
            self.status_var.set("Complete (no improvements)")


def main():
    root = tk.Tk()
    app = EvolutionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
