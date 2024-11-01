import numpy as np

from rec_sys import cf_algorithms_to_complete as cf

if __name__ == "__main__":
    matrix_a = [
        [4, 5, 0, 5, 1, 0, 3, 2],
        [0, 3, 4, 3, 1, 2, 1, 0],
        [2, 0, 1, 3, 0, 4, 5, 3],
    ]

    print("a")

    # AB
    print(f"AB: {1 - cf.cosine_sim(matrix_a[0], matrix_a[1])}")

    # AC
    print(f"AC: {1 - cf.cosine_sim(matrix_a[0], matrix_a[2])}")

    # BC
    print(f"BC: {1 - cf.cosine_sim(matrix_a[1], matrix_a[2])}")

    print("b")

    matrix_b = [
        [2, 3, 0, 3, 1, 0, 1, 2],
        [0, 1, 2, 1, 1, 2, 1, 0],
        [2, 0, 1, 1, 0, 2, 3, 1],
    ]

    # AB
    print(f"AB: {1 - cf.cosine_sim(matrix_b[0], matrix_b[1])}")

    # AC
    print(f"AC: {1 - cf.cosine_sim(matrix_b[0], matrix_b[2])}")

    # BC
    print(f"BC: {1 - cf.cosine_sim(matrix_b[1], matrix_b[2])}")

    print("c")

    matrix_c = np.array(
        [[4, 5, 0, 5, 1, 0, 3, 2], [0, 3, 4, 3, 1, 2, 1, 0], [2, 0, 1, 3, 0, 4, 5, 3]],
        dtype=float,
    )

    def normalize_ratings(matrix):
        normalized_matrix = matrix.copy()
        for i in range(matrix.shape[0]):
            user_ratings = matrix[i]
            # Calculate average of non-zero (non-blank) entries
            non_zero_ratings = user_ratings[user_ratings != 0]
            if len(non_zero_ratings) > 0:
                user_mean = np.mean(non_zero_ratings)
                # Subtract user mean from non-zero entries
                normalized_matrix[i, user_ratings != 0] -= user_mean
        return normalized_matrix

    normalized_matrix_c = normalize_ratings(matrix_c)

    # AB
    print(f"AB: {1 - cf.cosine_sim(normalized_matrix_c[0], normalized_matrix_c[1])}")

    # AC
    print(f"AC: {1 - cf.cosine_sim(normalized_matrix_c[0], normalized_matrix_c[2])}")

    # BC
    print(f"BC: {1 - cf.cosine_sim(normalized_matrix_c[1], normalized_matrix_c[2])}")

    print("d")

    matrix_d = np.array(
        [[4, 5, 0, 5, 1, 0, 3, 2], [0, 3, 4, 3, 1, 2, 1, 0], [2, 0, 1, 3, 0, 4, 5, 3]],
        dtype=float,
    )

    num_users = matrix_d.shape[0]

    # Calculate Pearson correlations between each pair of users
    for i in range(num_users):
        for j in range(i, num_users):
            if i == j:
                continue
            else:
                correlation = cf.pearson_correlation(matrix_d[i], matrix_d[j])
                print(f"Correlation between user {i} and user {j}: {correlation}")