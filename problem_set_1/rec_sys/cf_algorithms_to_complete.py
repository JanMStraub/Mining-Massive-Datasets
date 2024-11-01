# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import psutil
import os
import numpy as np
import scipy.sparse as sp
import shelve
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.sparse.linalg import norm as sparse_norm
from rec_sys.data_util import load_movielens_tf, print_df_stats


def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None


def center_and_nan_to_zero(matrix, axis=0):
    """Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    dot = np.dot(um_normalized.T, vector)
    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    print(
        f"\n>>> CF computation for UM w/ shape: "
        + f"{orig_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n"
    )

    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(
            np.isnan(orig_utility_matrix[item_index, :]) == False
        )[0]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(similarities[users_who_rated])
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[
            np.isnan(similarities[best_among_who_rated]) == False
        ]
        if best_among_who_rated.size > 0:
            sim_vals = similarities[best_among_who_rated]
            rating_neighbors = orig_utility_matrix[item_index, best_among_who_rated]

            # Compute the rating of the item
            rating_of_item = np.dot(sim_vals, rating_neighbors) / np.sum(
                np.abs(sim_vals)
            )
        else:
            rating_of_item = np.nan
        print(
            f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}"
        )
        return rating_of_item

    num_items = orig_utility_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


### Exercise 2 ###


def centered_cosine_sim(vec1, vec2):

    vec1_centered = np.nan_to_num(vec1 - np.nanmean(vec1))
    vec2_centered = np.nan_to_num(vec2 - np.nanmean(vec2))

    return cosine_sim(vec1_centered, vec2_centered)


def fast_centered_cosine_sim(sparse_mat, vec):
    vec_centered = np.nan_to_num(vec - np.nanmean(vec))

    # Compute row-wise mean and center the sparse matrix
    mat_centered = sparse_mat - sparse_mat.mean(axis=1)

    norms = np.linalg.norm(mat_centered, axis=0)
    um_normalized = mat_centered / norms

    # Compute the dot product of transposed normalized matrix and the vector
    dot = np.dot(um_normalized.T, vec_centered)
    return dot / np.linalg.norm(vec)


def pearson(sparse_mat, vec):
    # Center the vector
    vec_mean = vec.mean()
    centered_vec = vec - vec_mean

    # Calculate centered cosine similarity for each row
    centered_cosine_similarities = []
    for i in range(sparse_mat.shape[0]):
        # Extract row and center it
        row = sparse_mat.getrow(i).toarray().flatten()
        row_mean = row.mean()
        centered_row = row - row_mean

        # Compute dot product of centered row and centered vector
        dot_product = np.dot(centered_row, centered_vec)

        # Compute norms of centered row and centered vector
        row_norm = np.linalg.norm(centered_row)
        vec_norm = np.linalg.norm(centered_vec)

        # Calculate centered cosine similarity, handling division by zero
        similarity = dot_product / (row_norm * vec_norm + 1e-10)
        centered_cosine_similarities.append(similarity)

    # Resulting similarities
    return centered_cosine_similarities


### Exercise 3 ###


def center_and_nan_to_zero_sparse(matrix, axis=0):

    # Compute the mean along the specified axis, resulting in a 1D array
    mean_values = np.array(matrix.mean(axis=axis)).flatten()

    if axis == 0:
        # Subtract column means: create a matrix with means broadcasted along columns
        mean_matrix = sp.csr_matrix(
            np.repeat(mean_values, matrix.shape[0]).reshape(matrix.shape[1], -1)
        ).T
        centered_matrix = matrix - mean_matrix
    else:
        # Subtract row means: create a matrix with means broadcasted along rows
        mean_matrix = sp.csr_matrix(
            np.repeat(mean_values, matrix.shape[1]).reshape(matrix.shape[0], -1)
        )
        centered_matrix = matrix - mean_matrix

    # Replace NaNs with zero in sparse matrices (implicitly handled as zeros)
    return centered_matrix


def rate_all_items_sparse(orig_sparse_utility_matrix, user_index, neighborhood_size):
    print(
        f"\n>>> CF computation for sparse UM w/ shape: "
        + f"{orig_sparse_utility_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n"
    )

    # Center the sparse matrix
    clean_utility_matrix = center_and_nan_to_zero_sparse(orig_sparse_utility_matrix)
    user_col = clean_utility_matrix[:, user_index].toarray().flatten()

    # Calculate similarity between the user and all other users
    similarities = pearson(clean_utility_matrix, user_col)

    def rate_one_item(item_index):
        if not np.isnan(orig_sparse_utility_matrix[item_index, user_index]):
            return orig_sparse_utility_matrix[item_index, user_index]

        users_who_rated = orig_sparse_utility_matrix[item_index, :].nonzero()[1]
        sorted_similarities = np.argsort(similarities[users_who_rated])
        best_among_who_rated = users_who_rated[sorted_similarities[-neighborhood_size:]]

        # Remove any NaN similarities
        best_among_who_rated = best_among_who_rated[
            ~np.isnan(similarities[best_among_who_rated])
        ]

        if best_among_who_rated.size > 0:
            sim_vals = similarities[best_among_who_rated]
            rating_neighbors = (
                orig_sparse_utility_matrix[item_index, best_among_who_rated]
                .toarray()
                .flatten()
            )
            rating_of_item = np.dot(sim_vals, rating_neighbors) / np.sum(
                np.abs(sim_vals)
            )
        else:
            rating_of_item = np.nan

        print(
            f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}"
        )
        return rating_of_item

    num_items = orig_sparse_utility_matrix.shape[0]
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings


### Exercise 4 ###


def create_data_structures_with_shelve(
    config, rated_by_path="rated_by.db", user_col_path="user_col.db"
):

    # Load the dataset using the provided config
    ratings_tf, user_ids_voc, movie_ids_voc = load_movielens_tf(config)

    # Open shelves for persistent storage
    with shelve.open(rated_by_path, writeback=True) as rated_by, shelve.open(
        user_col_path, writeback=True
    ) as user_col:
        for rating in ratings_tf:
            user_id = int(user_ids_voc["user_id"].numpy())
            movie_id = int(movie_ids_voc["movie_id"].numpy())
            rating_value = float(rating["user_rating"].numpy())

            # Update rated_by dictionary in the shelf
            if movie_id not in rated_by:
                rated_by[movie_id] = []
            rated_by[movie_id].append((user_id, rating_value))

            # Update user_col dictionary in the shelf
            if user_id not in user_col:
                user_col[user_id] = []
            user_col[user_id].append((movie_id, rating_value))

        # Convert user_col to sparse format and store as tuples
        for user_id, movie_ratings in user_col.items():
            # Create sparse vector
            movie_ids, ratings = zip(*movie_ratings)
            sparse_vec = sp.csr_matrix(
                (ratings, (np.zeros(len(ratings)), movie_ids)),
                shape=(1, len(movie_ids)),
            )
            # Store as serializable tuple
            user_col[user_id] = (
                sparse_vec.data,
                sparse_vec.indices,
                sparse_vec.indptr,
                sparse_vec.shape,
            )

    print("Data structures rated_by and user_col created with Python shelves.")


def load_user_sparse_vector(user_id, user_col_path="user_col.db"):

    with shelve.open(user_col_path) as user_col:
        if user_id in user_col:
            data, indices, indptr, shape = user_col[user_id]
            return sp.csr_matrix((data, indices, indptr), shape=shape)
        else:
            return None


def load_movie_ratings(movie_id, rated_by_path="rated_by.db"):

    with shelve.open(rated_by_path) as rated_by:
        return rated_by.get(movie_id, [])


### Exercise 5 ###


def estimate_rating(user, item, utility_matrix, neighborhood_size=5):

    # Center and normalize the utility matrix
    mean_user_rating = np.nanmean(utility_matrix, axis=1).flatten()
    utility_matrix_centered = utility_matrix - mean_user_rating[:, None]

    # Compute similarity with other users or items (user-based in this case)
    similarities = pearson(
        utility_matrix_centered, utility_matrix[user, :].toarray().flatten()
    )

    # Find the neighborhood (top-N most similar users)
    neighbor_indices = np.argsort(-similarities)[:neighborhood_size]
    neighbor_ratings = utility_matrix[neighbor_indices, item].toarray().flatten()

    # Calculate weighted average of neighbor ratings
    weighted_sum = sum(
        similarities[i] * neighbor_ratings[i]
        for i in neighbor_indices
        if not np.isnan(neighbor_ratings[i])
    )
    similarity_sum = sum(
        similarities[i] for i in neighbor_indices if not np.isnan(neighbor_ratings[i])
    )

    # Predicted rating
    predicted_rating = mean_user_rating[user] + (
        weighted_sum / similarity_sum if similarity_sum != 0 else 0
    )
    return predicted_rating
