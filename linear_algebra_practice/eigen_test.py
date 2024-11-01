import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')
from eigen_value_vector import get_one_eigen_vector, get_eigens, get_eigen_values_2d_matrix

import numpy as np
import unittest
class EigenTests(unittest.TestCase):
    def test_one_eigen_vector_1(self):
        A = np.array([[4, 1],
              [2, 3]])
        actual = get_one_eigen_vector(A, 2, 1)
        expected = np.array([-0.4472136, 0.89442719])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_one_eigen_vector_2(self):
        A = np.array([[4, 1],
              [2, 3]])
        actual = get_one_eigen_vector(A, 2, 0)
        expected = np.array([-0.4472136, 0.89442719])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_one_eigen_vector_3(self):
        A = np.array([[4, 1],
              [2, 3]])
        actual = get_one_eigen_vector(A, 5, 1)
        expected = np.array([0.70710678, 0.70710678])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_one_eigen_vector_4(self):
        A = np.array([[4, 1],
              [2, 3]])
        actual = get_one_eigen_vector(A, 5, 0)
        expected = np.array([0.70710678, 0.70710678])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_get_eigens(self):
        A = np.array([[4, 1],
              [2, 3]])

        actual = get_eigens(A)

        np.testing.assert_equal(actual[0][0], 5.0, "First result's values are different")
        np.testing.assert_equal(actual[1][0], 2.0, "Second result's values are different")
        np.testing.assert_array_almost_equal(actual[0][1], np.array([0.70710678, 0.70710678]), 7, "First result's arrays are different", verbose=True)

        np.testing.assert_array_almost_equal(actual[1][1], np.array([-0.4472136 ,  0.89442719]), 7, "Second result's arrays are different", verbose=True)

    def test_get_eigen_values(self):
        A = np.array([1, -7, 10])
        actual = get_eigen_values_2d_matrix(A)
        np.testing.assert_array_almost_equal(actual, np.array([5., 2.]), 7, "Eigen values are different", verbose=True)