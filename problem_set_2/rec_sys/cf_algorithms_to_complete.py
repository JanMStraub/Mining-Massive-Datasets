# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np


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


def pearson_correlation(user1_ratings, user2_ratings):
    # Find items both users have rated
    common_items = ~np.isnan(user1_ratings) & ~np.isnan(user2_ratings)
    
    # If there are no common items, return a correlation of 0
    if not np.any(common_items):
        return 0
    
    # Extract the common ratings
    u1_common = user1_ratings[common_items]
    u2_common = user2_ratings[common_items]
    
    # Calculate mean ratings for each user on these common items
    mean1 = np.mean(u1_common)
    mean2 = np.mean(u2_common)
    
    # Calculate the Pearson correlation coefficient
    numerator = np.sum((u1_common - mean1) * (u2_common - mean2))
    denominator = np.sqrt(np.sum((u1_common - mean1) ** 2)) * np.sqrt(np.sum((u2_common - mean2) ** 2))
    
    # Handle the case of zero denominator (no variance in ratings)
    if denominator == 0:
        return 0
    
    return numerator / denominator
