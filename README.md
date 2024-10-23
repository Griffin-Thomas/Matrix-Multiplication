# Matrix Multiplication Performance Test (CPU vs GPU)

This project tests the performance of matrix multiplication in Python using both CPU and GPU. The goal is to measure the performance differences when running large matrix multiplications on a **Ryzen 7 5800X3D CPU** (using NumPy) and compare it with an **NVIDIA RTX 3080 12 GB GPU** (using CuPy or PyTorch).

## TODO
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
To perform the matrix multiplication on the CPU with $N = 4096$, run:
```bash
python matmul_cpu.py -N 4096
```

Note that if you don't provide an $N$ value, it will default to 4096 anyway.

### GPU Test (CuPy)
Similarly, to perform the matrix multiplication on the GPU using CuPy with $N = 25000$, run:
```bash
python matmul_gpu_cupy.py -N 25000
```

## Example CPU Test (NumPy) Output

For each test, the script will generate and multiply two fairly large square matrices (size **4096 x 4096**) and print the following:
- The number of floating-point operations (FLOP).
- The time it took to perform the multiplication.
- The performance in GFLOPS (Giga Floating-Point Operations Per Second).

Here’s an example output from the CPU test:
```
Two 4096x4096 square matrices have been randomly generated

137.44 GFLOP
Matrix multiplication completed in 0.19 seconds
734.97 GFLOPS
```

## Example GPU Test (CuPy) Output

For each test, the script will generate and multiply two extra large square matrices (size **25000 x 25000**). 

Note that this is much **larger** than the CPU test. The reason why I chose $N = 25000$ instead is because I noticed the VRAM on my RTX 3080 12 GB would go up to **11 GB of utilization**. Therefore, if I went with some $N > 30000$ for example, the GPU would run out of memory and have issues with processing the matrix multiplication.

Since this GPU should be much faster than the CPU, I will measure with teraflops instead of gigaflops.

Here’s an example output from the GPU test:
```
Two 25000x25000 square matrices have been randomly generated

31.25 TFLOP
Matrix multiplication completed in 1.66 seconds
18.87 TFLOPS
```