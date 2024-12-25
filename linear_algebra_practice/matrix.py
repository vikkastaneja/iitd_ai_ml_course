# Steps to Perform Gaussian Elimination
# For each row:
# 1. Identify the Pivot: Start from the first column and look for the first non-zero element in that column from the top. This element is called the pivot.
# 2. Make Pivot 1: If the pivot isn’t 1, divide the entire row by the pivot value to make it 1.
# 3. Make Zeros Below the Pivot: Use row operations to make all elements below this pivot zero by subtracting a multiple of the pivot row from each row below it.
# 4. Move to the Next Row and Column: Repeat the steps for the next row and column, treating the next row's leading non-zero element as the pivot. Ensure each pivot is to the right of the previous row’s pivot.
#
# Matrix Rank:
# The rank of the matrix is the number of non-zero rows.

import numpy as np

def find_pivot_normalize_row(matrix, row):
    index = -1
    for i in range(len(matrix[0])):
        if matrix[row][i] != 0:
            index = i
            break

    if index == -1:
        return index

    element = matrix[row][index]

    # Divide the row by matrix[row][index]
    for i in range(index, len(matrix[0])):
        matrix[row][i] = matrix[row][i] / element

    return index

def reduce_row(matrix, current_row, row_to_reduce, pivot_index):
    if current_row == len(matrix) - 1 or pivot_index == len(matrix[0]):
        return

    times = matrix[row_to_reduce][pivot_index] / matrix[current_row][pivot_index]
    for i in range(pivot_index, len(matrix[0])):
        matrix[row_to_reduce][i] = matrix[row_to_reduce][i] - matrix[current_row][i] * times

def reduce_matrix(matrix):
    for i in range(len(matrix)):
        index = find_pivot_normalize_row(matrix, i)
        if (index == -1):
            continue

        for j in range(i+1, len(matrix)):
            reduce_row(matrix, i, j, index)

def count_zero_rows(matrix):
    count = 0
    for row in matrix:
        if all(element == 0 for element in row):
            count += 1
    return count

def find_rank_2d_matrix(matrix):
    temp_matrix = [row[:] for row in matrix]
    reduce_matrix(temp_matrix)
    return len(temp_matrix) - count_zero_rows(temp_matrix)

# Ensure that each row represents a feature and the matrix is numpy array
# This method uses transpose method to calculate covariance matrix
def get_covariance_matrix_2d_array_using_transpose(matrix):
    n = matrix.shape[1]
    sums = np.sum(matrix, axis=1)
    means = sums/n

    centered_matrix = (matrix.transpose() - means).transpose()
    covariance_matrix = centered_matrix.dot(centered_matrix.transpose()) / (n - 1)
    return covariance_matrix

def get_covariance(matrix, row_i, row_j):
    cov = sum(matrix[row_i] * matrix[row_j]) / (matrix.shape[1] - 1)
    return cov


# Ensure that each row represents a feature and the matrix is numpy array
# This method uses calculation of covariances using mathematical formula
def get_covariance_matrix_2d_array_using_cov_calc(matrix):
    num_records = matrix.shape[1]
    num_features = matrix.shape[0]

    sums = np.sum(matrix, axis=1)
    means = sums/num_records

    centered_matrix = (matrix.transpose() - means).transpose()
    covariance_matrix = []
    for i in range(centered_matrix.shape[0]):
        row = []
        for j in range(centered_matrix.shape[0]):
            row.append(get_covariance(centered_matrix, i, j))

        covariance_matrix.append(row)

    np_covariance_matrix = np.array(covariance_matrix)
    return np_covariance_matrix

def get_inverse_2x2_matrix(A):
    # Inverse of a 2x2 matrix exist, or a matrix is invertible if det(A) != 0.
    # Hence return None if det(A) = 0
    det_matrix = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if det_matrix == 0:
        return None

    adj_matrix = np.array([[A[1][1], -A[0][1]],[-A[1][0], A[0][0]]])

    return adj_matrix/det_matrix
