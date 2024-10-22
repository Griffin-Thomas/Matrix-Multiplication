import numpy as np
import time

# Size of NxN square matrices
N = 4096

if __name__ == "__main__":
    # Generate two random NxN matrices A and B with float32 precision
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    
    # Multiplying two NxN matrices involves NxN dot products, where each dot product
    # consists of N multiplications and N - 1 additions, which gives N^2 * N operations
    # for multiplication and the same for addition.
    # Therefore, the total number of floating-point operations (FLOP) is 2 * N^3
    flop = 2 * N * N * N
    
    # Converting to GFLOP is straightforward
    print(f"{flop / 1e9 :.2f} GFLOP")
    
    # We use time.monotonic() because it is a monotonic clock, meaning it is guaranteed never to go backward
    # This makes it ideal for measuring elapsed time in a reliable way, 
    # unaffected by system clock adjustments (e.g. due to daylight savings or manual changes).
    # Start the timer
    st = time.monotonic()
    
    # Perform the matrix multiplication
    C = A @ B
    
    # End the timer
    et = time.monotonic()
    
    # Calculate the time duration for the matrix multiplication
    s = et - st
    print(f"Matrix multiplication completed in {s:.2f} seconds")
    
    # Calculate the performance in GFLOPS (giga floating-point operations per second)
    print(f"{flop / s * 1e-9 :.2f} GFLOPS")