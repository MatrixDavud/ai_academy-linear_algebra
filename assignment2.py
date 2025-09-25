# ============================================
# Math4AI - Programming Assignment 2
# Systems of Linear Equations & Model Fitting
# ============================================

# --- Imports ---
import numpy as np

# --- Example System ---
A = [
    [2, 1, 3],
    [4, 4, 7],
    [2, 5, 9]
]

b = [1, 1, 3]

# =====================================================
# PART 2.1: Gaussian Elimination from Scratch
# =====================================================

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Returns:
        - list: solution vector x if unique solution exists
        - str: 'No solution' if inconsistent
        - str: 'Infinite solutions' if system has free variables
    """
    # --- Input validation ---
    if not A or not b:
        raise ValueError("Matrix A and vector b cannot be empty.")
    n = len(A)

    # Check A is square
    if any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square (n x n).")

    # Check dimensions
    if len(b) != n:
        raise ValueError("Length of b must match number of rows in A.")

    # Check that all elements are numeric
    for row in A:
        for val in row:
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError("All elements of matrix A must be numbers.")
    for val in b:
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise TypeError("All elements of vector b must be numbers.")

    # TODO: Convert A and b into augmented matrix form
    aug = [[float(val) for val in row] + [float(b[i])] for i, row in enumerate(A)]

    # TODO: Implement forward elimination (with pivoting)
    for i in range(n):
        # Find pivot row
        max_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[max_row][i]) < 1e-12:  # pivot ~ 0
            continue  # free variable / singular
        if max_row != i:
            aug[i], aug[max_row] = aug[max_row], aug[i]

        # Eliminate below pivot
        for j in range(i + 1, n):
            if abs(aug[i][i]) < 1e-12:
                continue
            factor = aug[j][i] / aug[i][i]
            for k in range(i, n + 1):
                aug[j][k] -= factor * aug[i][k]

    # TODO: Detect and handle division by zero via row swaps
    # (Handled above via partial pivoting)

    # TODO: Detect 'No solution' and 'Infinite solutions' cases
    rank = 0
    for row in aug:
        if any(abs(val) > 1e-12 for val in row[:-1]):
            rank += 1
        elif abs(row[-1]) > 1e-12:  # 0 = nonzero → inconsistent
            return "No solution"
    if rank < n:
        return "Infinite solutions"

    # TODO: Implement back substitution for unique solution case
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            return "Infinite solutions"
        rhs = aug[i][-1] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = rhs / aug[i][i]

    return x

# --- Solve using your function ---
solution_scratch = gaussian_elimination(A, b)
print("Solution (from scratch):", solution_scratch)


# =====================================================
# PART 2.2: NumPy Verification
# =====================================================

# Convert to NumPy arrays
np_A = np.array(A, dtype=float)
np_b = np.array(b, dtype=float)

try:
    np_solution = np.linalg.solve(np_A, np_b)
    print("Solution (NumPy):", np_solution)
except np.linalg.LinAlgError as e:
    print("NumPy could not solve the system:", e)


# =====================================================
# Verification
# =====================================================
# TODO: Compare scratch implementation result with NumPy result if unique
if isinstance(solution_scratch, list) or isinstance(solution_scratch, np.ndarray):
    # TODO: Check closeness between the two solutions
    if 'np_solution' in locals():
        print("Scratch vs NumPy close?:", np.allclose(solution_scratch, np_solution))
