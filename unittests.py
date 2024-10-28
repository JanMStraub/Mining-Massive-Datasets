import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine as scipy_cosine

from rec_sys.cf_algorithms_to_complete import (
    centered_cosine_sim,
    fast_centered_cosine_sim,
    pearson,
)


def scipy_centered_cosine_sim_vec(mat, vec):
    """Compute cosine similarity on centered vectors using scipy's cosine function."""
    vec_centered = np.nan_to_num(vec - np.nanmean(vec))
    mat_centered = np.nan_to_num(mat - np.nanmean(mat))
    # scipy's cosine returns 1 - cosine similarity, so we subtract from 1
    return 1 - scipy_cosine(mat_centered, vec_centered)


def scipy_centered_cosine_sim_mat(mat, vec):
    """
    Compute cosine similarity on centered vectors between each row of a sparse matrix and a vector.
    """
    # Center the dense vector by subtracting its mean
    vec_mean = np.nanmean(vec)
    vec_centered = vec - vec_mean
    
    # Prepare an array to store cosine similarities
    centered_cosine_similarities = []

    # Process each row in the sparse matrix
    for i in range(mat.shape[0]):
        # Convert row to dense format and flatten to 1D
        row = mat.getrow(i).toarray().flatten()
        
        # Center the row by subtracting the mean
        row_mean = row.mean()  # Or use row[row != 0].mean() to exclude zeros from mean
        row_centered = row - row_mean
        
        # Calculate cosine similarity between the centered row and centered vector
        similarity = 1 - scipy_cosine(row_centered, vec_centered)
        centered_cosine_similarities.append(similarity)

    return np.array(centered_cosine_similarities)


def to_sparse_matrix(matrix):
    """Returns a sparse matrix, given a dense matrix or vector"""
    # Ensure the input is a numpy array
    matrix = np.asarray(matrix)

    # Identify non-NaN indices and values
    non_nan_indices = np.where(~np.isnan(matrix))
    non_nan_values = matrix[non_nan_indices]

    # Check if the input is a 1D vector
    if matrix.ndim == 1:
        # Treat as a single-row matrix with shape (1, len(matrix))
        sparse_matrix = csr_matrix(
            (
                non_nan_values,
                (np.zeros(len(non_nan_values), dtype=int), non_nan_indices[0]),
            ),
            shape=(1, matrix.shape[0]),
        )
    else:
        # Handle as a regular 2D matrix
        sparse_matrix = csr_matrix(
            (non_nan_values, non_nan_indices), shape=matrix.shape
        )

    return sparse_matrix


class TestCosineSimilarity(unittest.TestCase):

    def test_centered_cosine_sim_b1(self):
        # Test with k = 100 and xi = i + 1
        k = 100
        vec_x = np.array([i + 1 for i in range(k)])  # [1, 2, ..., 100]
        vec_y = np.array([vec_x[k - 1 - j] for j in range(k)])  # [100, 99, ..., 1]

        expected_similarity = scipy_centered_cosine_sim_vec(vec_x, vec_y)
        similarity = centered_cosine_sim(vec_x, vec_y)

        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)

    def test_centered_cosine_sim_b2(self):
        # Test with k = 100 and handling NaNs
        k = 100
        vec_x = []
        vec_y = []
        c_values = [2, 3, 4, 5, 6]

        # Create vec_x
        for i in range(k):
            if any(i in range(c, c + 100, 10) for c in c_values):
                xi = np.nan
            else:
                xi = i + 1
            vec_x.append(xi)

        # Create vec_y based on vec_x
        for j in range(k):
            i = k - 1 - j
            if 0 <= i < len(vec_x):
                vec_y.append(vec_x[i])
            else:
                vec_y.append(np.nan)

        vec_x = np.array(vec_x)
        vec_y = np.array(vec_y)

        similarity = centered_cosine_sim(vec_x, vec_y)
        expected_similarity = scipy_centered_cosine_sim_vec(vec_x, vec_y)

        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)

    def test_fast_centered_cosine_sim(self):
        # Sparse matrix and a vec
        sparse_mat = csr_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        vec = np.array([1, 0, 1])

        # Calculate centered cosine similarity for sparse matrix and dense vec
        # similarity = fast_centered_cosine_sim(sparse_mat, vec)
        similarity = pearson(sparse_mat, vec)
        expected_similarity = scipy_centered_cosine_sim_mat(sparse_mat, vec)
        
        self.assertTrue(np.allclose(similarity, expected_similarity, atol=0.1))


if __name__ == "__main__":
    unittest.main()
