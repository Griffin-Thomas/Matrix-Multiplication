import cupy as cp
import time
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Matrix multiplication performance test.")
parser.add_argument('-N', type=int, default=25000, help='Size of the matrix (default: 25000)')
args = parser.parse_args()

# Use the passed value of N or default to 25000
# Size of the NxN square matrix
N = args.N

if __name__ == "__main__":
    # Generate two random NxN matrices A and B with float32 precision
    A = cp.random.randn(N, N).astype(cp.float32)
    B = cp.random.randn(N, N).astype(cp.float32)
    
    # Multiplying two NxN matrices involves NxN dot products, where each dot product
    # consists of N multiplications and N - 1 additions, which gives N^2 * N operations
    # for multiplication and the same for addition.
    # Therefore, the total number of floating-point operations (FLOP) is 2 * N^3
    flop = 2 * N * N * N
    
    # Converting to TFLOP is straightforward
    print(f"{flop / 1e12 :.2f} TFLOP")
    
    # We use time.monotonic() because it is a monotonic clock, meaning it is guaranteed never to go backward
    # This makes it ideal for measuring elapsed time in a reliable way, 
    # unaffected by system clock adjustments (e.g. due to daylight savings or manual changes).
    # Start the timer
    st = time.monotonic()
    
    # Perform the matrix multiplication
    C = A @ B
    cp.cuda.Stream.null.synchronize() # Ensure GPU operations finish
    
    # End the timer
    et = time.monotonic()
    
    # Calculate the time duration for the matrix multiplication
    s = et - st
    print(f"Matrix multiplication completed in {s:.2f} seconds")
    
    # Calculate the performance in TFLOPS (tera floating-point operations per second)
    print(f"{flop / s * 1e-12 :.2f} TFLOPS")