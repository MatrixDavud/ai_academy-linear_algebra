# ============================================
# Math4AI - Programming Assignment 1
# Vector Operations & Semantic Similarity
# ============================================

# --- Imports ---
import numpy as np

# --- Sample Word Vectors (3D Example) ---
v_king   = [0.8, 0.65, 0.0]
v_man    = [0.6, 0.4,  0.0]
v_woman  = [0.7, 0.3,  0.2]
v_queen  = [0.9, 0.55, 0.2]

# =====================================================
# PART 1.1: King - Man + Woman Analogy (From Scratch)
# =====================================================

def check_vector(u):
    """Raise error if vector is empty or contains non-numeric elements."""
    if not u:
        raise ValueError("Vector cannot be empty.")
    for val in u:
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise TypeError("All elements of vectors must be numbers.")

def check_same_length(u, v):
    """Raise ValueError if u and v have different lengths."""
    check_vector(u)
    check_vector(v)
    if len(u) != len(v):
        raise ValueError("Vectors must be of same length.")

def vector_add(u, v):
    """Add two vectors u and v."""
    check_same_length(u, v)
    return [u[i] + v[i] for i in range(len(u))]

def vector_sub(u, v):
    """Subtract vector v from u."""
    check_same_length(u, v)
    return [u[i] - v[i] for i in range(len(u))]

# --- Analogy computation ---
# v_result = king - man + woman
v_result = vector_add(vector_sub(v_king, v_man), v_woman)
print("Analogy result (from scratch):", v_result)


# =====================================================
# PART 1.2: Cosine Similarity (From Scratch)
# =====================================================

def dot_product(u, v):
    """Compute the dot product of u and v."""
    check_same_length(u, v)
    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result

def norm(u):
    """Compute the L2 norm of vector u."""
    check_vector(u)
    return (sum(x * x for x in u)) ** 0.5

def cosine_similarity(u, v):
    """Compute cosine similarity between u and v."""
    check_same_length(u, v)
    norm_u = norm(u)
    norm_v = norm(v)
    if norm_u == 0 or norm_v == 0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors.")
    return dot_product(u, v) / (norm_u * norm_v)

# --- Cosine similarity between analogy result & queen ---
similarity_scratch = cosine_similarity(v_result, v_queen)
print("Cosine similarity (from scratch):", similarity_scratch)


# =====================================================
# PART 1.3: NumPy Verification
# =====================================================

# Convert to numpy arrays
np_king   = np.array(v_king)
np_man    = np.array(v_man)
np_woman  = np.array(v_woman)
np_queen  = np.array(v_queen)

# --- Analogy computation with NumPy ---
np_result = np_king - np_man + np_woman
print("Analogy result (NumPy):", np_result)

# --- Cosine similarity with NumPy ---
similarity_numpy = np.dot(np_result, np_queen) / (np.linalg.norm(np_result) * np.linalg.norm(np_queen))
print("Cosine similarity (NumPy):", similarity_numpy)


# =====================================================
# Verification
# =====================================================
print("\n--- Verification ---")
print("Result equal (scratch vs NumPy)?", np.allclose(v_result, np_result))
print("Similarity close (scratch vs NumPy)?", np.isclose(similarity_scratch, similarity_numpy))
