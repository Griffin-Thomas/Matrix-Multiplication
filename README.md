# Matrix Multiplication Performance Test (CPU vs GPU)

This project tests the performance of matrix multiplication in Python using both CPU and GPU. The goal is to measure the performance differences when running large matrix multiplications on a **Ryzen 7 5800X3D CPU** (using NumPy) versus an **NVIDIA RTX 3080 12 GB GPU** (using CuPy or PyTorch).

## TODO
- [ ] `matmul_gpu_cupy.py`: A Python script that performs matrix multiplication using CuPy (GPU).
- [ ] `matmul_gpu_torch.py`: A Python script that performs matrix multiplication using PyTorch (GPU).

## Project Structure

- `matmul_cpu.py`: A Python script that performs matrix multiplication using NumPy (CPU).

## Prerequisites

Make sure **Python 3.12.7** is installed on your system along with the necessary libraries depending on whether you're testing on the CPU or GPU.
Oh, and have an **NVIDIA GPU with CUDA Toolkit 12.6** installed.

### Install Dependencies

For CPU testing with **NumPy**:
```bash
pip install numpy
```

For GPU testing with **CuPy**:
```bash
pip install cupy-cuda12x
```

### Verify CUDA Installation

Ensure that CUDA is properly installed and your system recognizes the GPU by running the following command:
```bash
nvidia-smi
```

If CUDA is properly installed, you'll see the details of your NVIDIA GPU. The output should look something like this:
```bash
Mon Oct 21 21:56:00 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080      WDDM  |   00000000:2B:00.0  On |                  N/A |
| 41%   33C    P8             37W /  400W |    2173MiB /  12288MiB |      6%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Running the Tests

### CPU Test (NumPy)
To perform the matrix multiplication on the CPU, run:
```bash
python matmul_cpu.py
```

### GPU Test (CuPy)
To perform the matrix multiplication on the GPU using CuPy, run:
```bash
python matmul_gpu_cupy.py
```

## Example Output

For each test, the script will generate and multiply two fairly large square matrices (size **4096 x 4096**) and print the following:
- The number of floating-point operations (FLOP).
- The time it took to perform the multiplication.
- The performance in GFLOPS (Giga Floating-Point Operations Per Second).

Here’s an example output from the CPU test:
```bash
137.44 GFLOP
Matrix multiplication completed in 0.19 seconds
734.97 GFLOPS
```