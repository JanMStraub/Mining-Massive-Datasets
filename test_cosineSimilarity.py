import unittest
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine #used for testing my own implementation of it to have something to compare against
from rec_sys.cf_algorithms_to_complete import (
    centered_cosine_sim,
    fast_centered_cosine_sim,
)
from scipy.sparse import csr_matrix

k = 100

# Case b.1: Create vector_x and vector_y where xi = i+1 and yj = xi with i+j = k-1
vector_x_b1 = np.array([i + 1 for i in range(k)])
vector_y_b1 = np.array([vector_x_b1[k - 1 - i] for i in range(k)])

# Case b.2: Create vector_x and vector_y with specific NaNs at [c, c+10, ...] in xi, c = [2,3,4,5,6]
nan_indices = []
for c in [2, 3, 4, 5, 6]:
    nan_indices.extend(range(c, k, 10))
vector_x_b2 = np.array([i + 1 if i not in nan_indices else np.nan for i in range(k)])
vector_y_b2 = np.array([vector_x_b2[k - 1 - i] for i in range(k)])

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
        sparse_matrix = csr_matrix((non_nan_values, (np.zeros(len(non_nan_values), dtype=int), non_nan_indices[0])), shape=(1, matrix.shape[0]))
    else:
        # Handle as a regular 2D matrix
        sparse_matrix = csr_matrix((non_nan_values, non_nan_indices), shape=matrix.shape)
    
    return sparse_matrix



def scipy_centered_cosine_sim(vec1, vec2):
    """Compute cosine similarity on centered vectors using scipy's cosine function."""
    centered_vec1 = vec1 - np.nanmean(vec1)
    centered_vec2 = vec2 - np.nanmean(vec2)
    
    # Replace NaNs with 0 in centered vectors
    centered_vec1 = np.nan_to_num(centered_vec1)
    centered_vec2 = np.nan_to_num(centered_vec2)
    
    # scipy's cosine returns 1 - cosine similarity, so we subtract from 1
    return 1 - scipy_cosine(centered_vec1, centered_vec2)


class TestCenteredCosineSim(unittest.TestCase):
    
    def test_centered_cosine_sim_b1(self):
        """Test centered_cosine_sim with vectors where xi = i + 1 and yj = xi (i + j = k - 1)"""
        # Compute similarity using our function
        similarity = centered_cosine_sim(vector_x_b1, vector_y_b1)
        # Compute similarity using scipy as a reference
        expected_similarity = scipy_centered_cosine_sim(vector_x_b1, vector_y_b1)
        
        print("Similarity (Case b.1):", similarity)
        print("Expected Similarity (Case b.1):", expected_similarity)
        
        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)

    def test_centered_cosine_sim_b2(self):
        """Test centered_cosine_sim with NaNs in vector_x and symmetric values in vector_y"""
        # Compute similarity using our function
        similarity = centered_cosine_sim(vector_x_b2, vector_y_b2)
        # Compute similarity using scipy as a reference
        expected_similarity = scipy_centered_cosine_sim(vector_x_b2, vector_y_b2)
        
        print("Similarity (Case b.2):", similarity)
        print("Expected Similarity (Case b.2):", expected_similarity)
        
        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)

    def test_fast_centered_cosine_sim_b1(self):
        """Test fast_centered_cosine_sim with sparse vectors using scipy as a benchmark"""

        # Convert vector_x_b1 and vector_y_b1 to sparse format for testing
        sparse_x = to_sparse_matrix(vector_x_b1)
        sparse_y = to_sparse_matrix(vector_y_b1)
        
        # Calculate similarity using our function
        similarity = fast_centered_cosine_sim(sparse_x, sparse_y)
        # Calculate expected similarity using scipy on dense data as a benchmark
        expected_similarity = scipy_centered_cosine_sim(vector_x_b1, vector_y_b1)
        
        print("Similarity (Sparse, Case b.1):", similarity)
        print("Expected Similarity (Sparse, Case b.1):", expected_similarity)
        
        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)

    def test_fast_centered_cosine_sim_b2(self):
        """Test fast_centered_cosine_sim with sparse vectors using scipy as a benchmark"""

        # Convert vector_x_b1 and vector_y_b1 to sparse format for testing
        sparse_x = to_sparse_matrix(vector_x_b2)
        sparse_y = to_sparse_matrix(vector_y_b2)
        
        # Calculate similarity using our function
        similarity = fast_centered_cosine_sim(sparse_x, sparse_y)
        # Calculate expected similarity using scipy on dense data as a benchmark
        expected_similarity = scipy_centered_cosine_sim(vector_x_b2, vector_y_b2)
        
        print("Similarity (Sparse, Case b.2):", similarity)
        print("Expected Similarity (Sparse, Case b.2):", expected_similarity)
        
        self.assertAlmostEqual(similarity, expected_similarity, delta=0.1)


# Run the tests
if __name__ == "__main__":
    unittest.main()
