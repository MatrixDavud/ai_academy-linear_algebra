# Math4AI: Linear Algebra - Programming Assignment 3
# Starter Code Template

import numpy as np
from scipy.linalg import lu as scipy_lu  # Used for verification

# --- Helper Function for Pretty Printing ---
def print_matrix(name, m):
    """
    Helper function to print a matrix with its name.
    Handles None for non-invertible matrices.
    """
    print(f"{name}:")
    if m is None:
        print("None (Matrix is singular or function not implemented)")
    else:
        # Set print options for better readability
        np.set_printoptions(precision=4, suppress=True)
        print(m)
    print("-" * 30)

# --- Problem Setup ---
# The matrix A for this assignment
A = np.array([
    [2., 1., 3.],
    [4., 4., 7.],
    [2., 5., 9.]
])

print_matrix("Original Matrix A", A)

# ====================================================================
# Part 3.1: Matrix Inverse via Gauss-Jordan Elimination
# ====================================================================

def invert_matrix(A):
    """
    Computes the inverse of a square matrix A using Gauss-Jordan elimination.
    
    Args:
        A (np.ndarray): A square numpy array.
        
    Returns:
        np.ndarray: The inverse of A, or None if A is singular.
    """
    A = A.astype(float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    # Create augmented matrix
    identity = np.identity(n)
    augmented_A = np.hstack((A, identity))
    print("Initial Augmented Matrix [A|I]:")
    print(augmented_A)
    print("\nStarting Gauss-Jordan Elimination...")

    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        pivot = augmented_A[i, i]
        if abs(pivot) < 1e-12:
            # Pivot too small, matrix is singular
            return None
        
        # Normalize pivot row
        augmented_A[i] = augmented_A[i] / pivot

        # Eliminate all other entries in column i
        for j in range(n):
            if j != i:
                factor = augmented_A[j, i]
                augmented_A[j] -= factor * augmented_A[i]

    # Extract inverse from right half
    inverse_A = augmented_A[:, n:]
    return inverse_A


# ====================================================================
# Part 3.2: LU Decomposition from Scratch
# ====================================================================

def lu_decomposition(A):
    """
    Performs LU decomposition of a square matrix A using Doolittle's algorithm.
    
    Args:
        A (np.ndarray): A square numpy array.
        
    Returns:
        (np.ndarray, np.ndarray): A tuple of (L, U) matrices.
    """
    A = A.astype(float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    L = np.identity(n)
    U = np.zeros((n, n))

    for i in range(n):
        # Compute row i of U
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # Compute column i of L
        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError("Zero pivot encountered, LU decomposition fails without pivoting.")
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

# --- Calling the function for Part 3.2 ---
print("--- Part 3.2: LU Decomposition from Scratch ---")
L_scratch, U_scratch = lu_decomposition(A.copy())
print_matrix("L (from scratch)", L_scratch)
print_matrix("U (from scratch)", U_scratch)


# ====================================================================
# Part 3.3: NumPy Verification
# ====================================================================
print("--- Part 3.3: NumPy Verification ---")

# 1. Verifying the Matrix Inverse
print("Verifying Matrix Inverse...")
A_inv_numpy = np.linalg.inv(A)
print_matrix("Inverse A (NumPy)", A_inv_numpy)

# 2. Verifying the LU Decomposition
print("Verifying LU Decomposition...")
# We check by multiplying L and U and see if we get back A
if L_scratch is not None and U_scratch is not None:
    product_LU = L_scratch @ U_scratch
    print_matrix("L @ U (from scratch)", product_LU)
    print_matrix("Original A (for comparison)", A)
    
    # A programmatic check for correctness
    is_correct = np.allclose(A, product_LU)
    print(f"Verification Check (A == L @ U): {is_correct}\n")
else:
    print("LU decomposition not yet implemented.\n")
    
# Optional: Compare with SciPy's LU decomposition
print("--- Comparing with SciPy LU Decomposition ---")
P, L_scipy, U_scipy = scipy_lu(A)
print_matrix("P (Permutation Matrix)", P)
print_matrix("L (SciPy)", L_scipy)
print_matrix("U (SciPy)", U_scipy)

# Verify that P @ L @ U == A
is_scipy_correct = np.allclose(A, P @ L_scipy @ U_scipy)
print(f"Verification with SciPy (P@L@U == A): {is_scipy_correct}\n")
