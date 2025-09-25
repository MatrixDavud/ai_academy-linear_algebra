# Math4AI: Linear Algebra - Programming Assignment 5
# Starter Code Template

import numpy as np
from scipy.linalg import null_space, orth

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_vectors(name, vecs):
    """Prints a list of basis vectors from a list of arrays or a 2D array."""
    print(f"{name}:")
    if vecs is None or len(vecs) == 0:
        print("[] (or not implemented)")
    elif isinstance(vecs, list) and all(isinstance(v, np.ndarray) for v in vecs):
        for i, v in enumerate(vecs):
            print(f"  Basis Vector {i+1}:\n{v.reshape(-1, 1)}")
    elif isinstance(vecs, np.ndarray) and vecs.ndim == 2:
         for i in range(vecs.shape[1]):
             print(f"  Basis Vector {i+1}:\n{vecs[:, i].reshape(-1, 1)}")
    else:
        print("Unsupported format for printing vectors.")
    print("-" * 40)



A1 = np.array([
    [1., 2., 3., 5.],
    [2., 4., 8., 12.],
    [3., 6., 7., 13.]
])
print_matrix("Matrix A for Part 1", A1)


# --- Reusable Helper Function (Students must implement) ---
def to_rref(M):
    """
    Converts a matrix M to its Reduced Row Echelon Form (RREF).
    
    Args:
        M (np.ndarray): The input matrix.
        
    Returns:
        np.ndarray: The RREF of M.
    """
    # 1. Make a copy to avoid modifying the original
    A = M.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0

    # 2. Loop through columns to find pivots
    for pivot_col in range(cols):
        if pivot_row >= rows:
            break
        
        # 3a. Find the row with a non-zero entry in pivot_col at or below pivot_row
        max_row = None
        for r in range(pivot_row, rows):
            if abs(A[r, pivot_col]) > 1e-12:  # tolerance for floating point
                max_row = r
                break
        
        if max_row is None:
            continue  # no pivot in this column, move to next column
        
        # 3b. Swap current row with max_row if needed
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        
        # 3c. Normalize pivot row so pivot element becomes 1
        pivot_val = A[pivot_row, pivot_col]
        A[pivot_row] = A[pivot_row] / pivot_val
        
        # 3d. Eliminate entries in the same column (both above and below)
        for r in range(rows):
            if r != pivot_row and abs(A[r, pivot_col]) > 1e-12:
                A[r] -= A[r, pivot_col] * A[pivot_row]
        
        pivot_row += 1

    return np.round(A, 10)  # rounding for cleaner output


# --- 5.1: Bases for the Four Subspaces ---

def find_column_space_basis(A):
    """Basis for C(A) from pivot columns of original A."""
    # --- YOUR CODE HERE ---
    # 1. Get RREF of A.
    # 2. Identify pivot column indices.
    # 3. Return the corresponding columns from the *original* matrix A.
    rref_A = to_rref(A)
    tol = 1e-12
    pivot_cols = []

    # For each row of the RREF, the first non-zero entry marks the pivot column.
    for i in range(rref_A.shape[0]):
        nz = np.where(np.abs(rref_A[i, :]) > tol)[0]
        if nz.size > 0:
            pivot_cols.append(int(nz[0]))

    # Deduplicate & sort (just in case)
    pivot_cols = sorted(list(dict.fromkeys(pivot_cols)))

    # If there are no pivots (zero matrix), return an empty (m x 0) array
    if len(pivot_cols) == 0:
        return np.zeros((A.shape[0], 0))

    # Return columns of the original matrix A corresponding to pivot columns
    return A[:, pivot_cols].copy()

def find_null_space_basis(A):
    """
    Finds the basis for the nullspace of matrix A.
    
    Args:
        A (np.ndarray): The input matrix.
        
    Returns:
        list: A list of numpy arrays, where each array is a basis vector for the nullspace.
    """
    m, n = A.shape
    
    # 1. Compute the RREF of A
    rref_A = to_rref(A)
    print_matrix("RREF of A", rref_A)
    
    basis_vectors = []
    
    # 2. Identify pivot and free columns
    pivot_cols = []
    row, col = 0, 0
    while row < m and col < n:
        if abs(rref_A[row, col] - 1) < 1e-12 and all(abs(rref_A[row2, col]) < 1e-12 for row2 in range(m) if row2 != row):
            pivot_cols.append(col)
            row += 1
        col += 1
    free_cols = [j for j in range(n) if j not in pivot_cols]
    
    # 3. Construct special solutions for each free variable
    for free in free_cols:
        # a. Create a vector of zeros
        vec = np.zeros(n)
        # b. Set the free variable to 1
        vec[free] = 1
        # c/d. Solve for pivot variables using RREF
        for i, pc in enumerate(pivot_cols):
            vec[pc] = -rref_A[i, free]
        # 4. Append to basis
        basis_vectors.append(vec)
    
    return basis_vectors


def find_row_space_basis(A):
    """Basis for C(A^T) from non-zero rows of RREF of A."""
    # --- YOUR CODE HERE ---
    # 1. Get RREF of A.
    # 2. The non-zero rows of the RREF form the basis for the row space.
    # 3. Return these rows as a list of vectors.

    rref_A = to_rref(A)
    tol = 1e-12
    row_basis = []
    for i in range(rref_A.shape[0]):
        row = rref_A[i, :].copy()
        if np.linalg.norm(row) > tol:
            row_basis.append(row)
    return row_basis

def find_left_null_space_basis(A):
    """Basis for N(A^T) is the nullspace of A^T."""
    # Hint: You can just use your existing nullspace function!
    print("Finding Left Nullspace by finding Nullspace of A.T")
    return find_null_space_basis(A.T)


print("\n--- 1.1: Finding the Bases ---")
C_A_basis = find_column_space_basis(A1.copy())
N_A_basis = find_null_space_basis(A1.copy())
C_AT_basis = find_row_space_basis(A1.copy())
N_AT_basis = find_left_null_space_basis(A1.copy())

print_vectors("Column Space Basis C(A)", C_A_basis)
print_vectors("Nullspace Basis N(A)", N_A_basis)
print_vectors("Row Space Basis C(A^T)", C_AT_basis)
print_vectors("Left Nullspace Basis N(A^T)", N_AT_basis)


# --- 5.2: Verification & The Fundamental Theorem ---
print("\n--- 1.2: Verification & The Fundamental Theorem ---")

# 1. Verify Dimensions
print("--- Verifying Dimensions ---")
if all(b is not None for b in [C_A_basis, N_A_basis, C_AT_basis, N_AT_basis]):
    rank_A = C_A_basis.shape[1]
    dim_N_A = len(N_A_basis)
    dim_C_AT = len(C_AT_basis)
    dim_N_AT = len(N_AT_basis)
    m, n = A1.shape

    print(f"Dimensions of A: {m}x{n}")
    print(f"dim(C(A)) = {rank_A} (Rank)")
    print(f"dim(C(A^T)) = {dim_C_AT} (Rank)")
    print(f"Are ranks equal? {rank_A == dim_C_AT}\n")

    print("Rank-Nullity Theorem for A:")
    print(f"dim(C(A)) + dim(N(A)) = {rank_A} + {dim_N_A} = {rank_A + dim_N_A}")
    print(f"Number of columns (n) = {n}")
    print(f"Is theorem satisfied? {rank_A + dim_N_A == n}\n")
    
    print("Rank-Nullity Theorem for A^T:")
    print(f"dim(C(A^T)) + dim(N(A^T)) = {dim_C_AT} + {dim_N_AT} = {dim_C_AT + dim_N_AT}")
    print(f"Number of rows (m) = {m}")
    print(f"Is theorem satisfied? {dim_C_AT + dim_N_AT == m}\n")
else:
    print("Bases not implemented, cannot verify dimensions.")

# 2. Verify Orthogonality
print("--- Verifying Orthogonality ---")
if all(b is not None for b in [C_A_basis, N_A_basis, C_AT_basis, N_AT_basis]):
    # Row Space _|_ Nullspace
    row_vec_1 = C_AT_basis[0]
    null_vec_1 = N_A_basis[0]
    dot_product_1 = np.dot(row_vec_1, null_vec_1)
    print(f"Dot product of a row space vector and a nullspace vector:")
    print(f"Result = {dot_product_1:.4f} (should be 0)")
    
    # Column Space _|_ Left Nullspace
    col_vec_1 = C_A_basis[:, 0]
    left_null_vec_1 = N_AT_basis[0]
    dot_product_2 = np.dot(col_vec_1, left_null_vec_1)
    print(f"\nDot product of a column space vector and a left nullspace vector:")
    print(f"Result = {dot_product_2:.4f} (should be 0)")
else:
    print("Bases not implemented, cannot verify orthogonality.")
