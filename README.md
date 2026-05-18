# Evaluating Expressions with GPU (Genetic Programming)

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white" />
  <img alt="Benchmark" src="https://img.shields.io/badge/Benchmark-CPU%20vs%20GPU-FF6F00" />
  <img alt="License" src="https://img.shields.io/badge/License-Not%20specified-lightgrey" />
</p>

## Overview

This project explores **Genetic Programming (GP)**, a variant of Genetic Algorithms where each individual is represented as an **expression tree (program)** instead of an integer array.

The goal is to evaluate many candidate expressions against a dataset and find the one that minimizes the **Mean Squared Error (MSE)** between predicted and target values.

Because program evaluation is expensive but highly parallelizable, this project focuses on a **GPU-based implementation** of the sequential baseline.

## Task

Implement a GPU version of the sequential evaluator (`sequential.py` in the assignment context) that:

1. Reads a CSV file:
   - Input features: all columns except the last
   - Target value: last column
2. Reads a `functions.txt` file containing candidate functions/expressions to evaluate
3. Computes predictions and MSE for each candidate expression
4. Finds the expression with the lowest MSE

## Parallelization Goals

The GPU implementation should evaluate both:

1. **Task Parallelism**: evaluate multiple candidate expressions in parallel
2. **Data Parallelism**: evaluate rows/samples of the dataset in parallel

Additionally:

- Push as much work as possible to the GPU
- Consider **dynamic kernel generation** from `functions.txt` (build kernel source as a string)
- Optionally modify `generate_inputs.py` to scale workload up/down for experiments

## Repository Structure

- `generate_inputs.py`: dataset/workload generator
- `sequential_parallel_benchmark.py`: benchmark and comparison utilities
- `codelab_benchmark.ipynb`: notebook for experimentation/analysis
- `report.pdf`: project report

## Suggested Workflow

1. Generate or prepare input data (`generate_inputs.py`)
2. Run baseline CPU evaluation (sequential)
3. Run GPU evaluation
4. Benchmark both versions for multiple combinations of:
   - Number of functions
   - Number of CSV rows
5. Record speedups, overhead, and crossover point (GPU becomes worthwhile)

## Report Checklist

Your report should explicitly answer:

1. How did you parallelize your program?
2. How many kernels and memory copies did you use?
3. How did you minimize the number of kernels called?
4. How did you minimize the number of data transfers required?
5. How did you choose the number of threads and blocks (for each kernel)?
6. How did you use shared memory?
7. How did you handle branch divergence/conflicts?
8. For which combination of number of functions and CSV rows does GPU outperform CPU?
9. If integrated in a full GP loop (evaluation, tournament, crossover, mutation, etc.), how would you adapt the code to reduce overhead across iterations?

## Practical Optimization Notes

- Fuse operations where possible (fewer kernels)
- Keep intermediate buffers on device across steps
- Reuse allocated device memory
- Batch evaluations to amortize launch overhead
- Prefer coalesced memory access patterns
- Use shared memory only when reuse justifies it
- Profile regularly (kernel time vs transfer time)

## How to Run

Adjust commands to your actual scripts and environment:

```bash
# Generate inputs (example)
python generate_inputs.py

# Run benchmark/experiments
python sequential_parallel_benchmark.py
```

## Deliverables

- GPU implementation of expression evaluation
- Benchmark evidence (CPU vs GPU)
- Final report addressing all checklist questions

---

If you want, I can also create a second version of this README with your exact course format (e.g., Objectives, Methodology, Results, Conclusions) so it matches your report structure one-to-one.
