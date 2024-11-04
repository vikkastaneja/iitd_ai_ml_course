# Steps of calculating principle component analysis dimensionality reduction
# 1. Perform mean centering
# 2. Calculate the covariance matrix
# 3. Calculate Eigen Value and Vector of covariance matrix
# 4. Compute Principal components and reduce the dimension of the data set based on higher variance dataset around Eigen Value, which is usually higher absolute value eigen value
import numpy as np

def get_pca_array_2d_matrix(row_based_features):
    # Create Covariance matrix
    from matrix import get_covariance_matrix_2d_array_using_cov_calc as get_covariance_matrix
    cov_matrix = get_covariance_matrix(row_based_features)

    # Create Eigen Vectors and Eigen Values
    from eigen_value_vector import get_eigens
    val_vec = get_eigens(cov_matrix)
    print(f"Eigen Values: {val_vec[0][0]} and {val_vec[1][0]}, Eigen Vectors: {val_vec[0][1]} and {val_vec[1][1]}")

    # Compute Principal components and reduce the dimension of the data set
    # 1. Sort eigen values (L1, L2) in descending order of magnitude
    # 2. Sort the eigen vectors (V1, V2) in the same order and create a matrix
    # 3. Two principal components are the two eigen vectors, with first one has the maximum variation of data.
    # 4. Sum of two eigen values = sum of two diagonal elements in eigen vector matrix
    # 5. Percentage of total variance for L1 = (L1/(L1 + L1))* 100
    #    Percentage of total variance for L2 = (L2/(L1 + L1))* 100
    # 6. Choose how many Vs are needed
    # 7. Transformation and reducing the dimension by matrix multiplication of original matrix with chosen eigen vector(s)
    # 8. Verify the amount of data loss by reconstructing and comparing the dataset with original one
    eigen_values = (val_vec[0][0], val_vec[1][0])
    eigen_vectors = (val_vec[0][1], val_vec[1][1])

    # Sort the eigen values in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    print(sorted_index)

    # Sort the eigen vectors in descending order of eigen values (first principal component based on the higher eigen value)
    sorted_vectors = []
    for i in sorted_index:
        sorted_vectors.append(eigen_vectors[i])
    print(sorted_vectors)

    # Validate that the higher variance eigen value is the first one
    per1 = 100*eigen_values[sorted_index[0]]/(eigen_values[sorted_index[0]] + eigen_values[sorted_index[1]])
    per2 = 100*eigen_values[sorted_index[1]]/(eigen_values[sorted_index[0]] + eigen_values[sorted_index[1]])
    print(f"Variance for eigen value {eigen_values[sorted_index[0]]} is {per1}%")
    print(f"Variance for eigen value {eigen_values[sorted_index[1]]} is {per2}%")

    # Based on eigen values, I will choose high eigen value only to have 1 dimension only
    # As 96% value is retained when I use first principal component
    eigen_subset = sorted_vectors[0].reshape(2,1)
    print(f"Eigen Vector corresponding to {eigen_values[sorted_index[0]]} is {eigen_subset}")

    # Reduce the dimension
    transformed = np.matmul(row_based_features.T, eigen_subset)
    print(row_based_features.shape)
    print(transformed.shape)
    print(eigen_subset.shape)

    # Reconstruct the original dataset
    orig = np.matmul(transformed, eigen_subset.T)
    print(orig)
    return (transformed, per1)

original = np.array([[2.5, 2.4],[0.5, 0.7],[2.2,2.9],[1.9,2.2],[3.1,3],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]])
print(get_pca_array_2d_matrix(original.T))