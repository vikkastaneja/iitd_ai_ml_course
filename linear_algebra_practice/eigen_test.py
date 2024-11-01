import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')
from eigen_value_vector import get_one_eigen_vector, get_eigens

import numpy as np
import unittest
class TryReflection(unittest.TestCase):
    def test_one_eigen_vector_1(self):
        actual = get_one_eigen_vector(np.array([4, 1]), 2, 1)
        expected = np.array([0.24253563, 0.9701425])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_one_eigen_vector_2(self):
        actual = get_one_eigen_vector(np.array([4, 1]), 2, 0)
        expected = np.array([-0.4472136, 0.89442719])

        np.testing.assert_array_almost_equal(actual, expected, 7, "Results are different", verbose=True)

    def test_get_eigens(self):
        A = np.array([[4, 1],
              [2, 3]])

        actual = get_eigens(A)
  
        np.testing.assert_equal(actual[0][0], 5.0, "First result's values are different")
        np.testing.assert_equal(actual[1][0], 2.0, "Second result's values are different")
        np.testing.assert_array_almost_equal(actual[0][1], np.array([0.70710678, 0.70710678]), 7, "First result's arrays are different", verbose=True)

        np.testing.assert_array_almost_equal(actual[1][1], np.array([-0.4472136 ,  0.89442719]), 7, "Second result's arrays are different", verbose=True)