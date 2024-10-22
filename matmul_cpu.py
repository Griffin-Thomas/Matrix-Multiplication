import numpy as np
import time

N = 4096

if __name__ == "__main__":
    # Generate two random NxN matrices A and B with float32 precision
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    
    # Multiplying two NxN matrices involves NxN dot products, where each dot product
    # consists of N multiplications and N - 1 additions, which gives N^2 * N operations
    # for multiplication and the same for addition
    # Therefore, the total number of FLOP is 2 * N^3
    flop = 2*N*N*N
    
    print(f"{flop / 1e9 :.2f} GFLOP")
    
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    
    s = et - st
    print(f"Matrix multiplication completed in {s:.2f} seconds")
    
    print(f"{flop / s * 1e-9 :.2f} GFLOPS")