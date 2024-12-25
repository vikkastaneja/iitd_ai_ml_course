import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')

from matrix import find_rank_2d_matrix as find_rank
from matrix import get_covariance_matrix_2d_array_using_transpose as get_covariance_transpose
from matrix import get_covariance_matrix_2d_array_using_cov_calc as get_covariance_cov_calc
from matrix import get_inverse_2x2_matrix
import pytest
import numpy as np

@pytest.fixture
def testdata():
    return [([[1, 2], [3, 4]], 2),
            ([[1, 2], [2, 4]], 1),
            ([[1, 2, 3], [2, 4, 8]], 2),
            ([[1]], 1),
            ([[0, 0, 0]], 0),
            ([[2, 3], [4, 5]], 2)]

def test_ranks(testdata):
    for data in testdata:
        assert find_rank(data[0]) == data[1]

@pytest.fixture
def covariance_test_data():
    return [
       (np.array([[7, 6, 1, 8, 3],
                  [6, 9, 9, 8, 8],
                  [8, 5, 2, 1, 6],
                  [3, 7, 9, 7, 2],
                  [3, 2, 2, 7, 3]]),
        np.array([[ 8.5,  -1.75,  1.,   -1.5,   3.75],
                [-1.75,  1.5,  -2.25,  2.5,  -0.5 ],
                [ 1.,   -2.25,  8.3,  -6.8,  -2.95],
                [-1.5,   2.5,  -6.8,   8.8,   0.2 ],
                [ 3.75, -0.5,  -2.95,  0.2,   4.3 ]])),
            (np.array([[1,2,3,4,5],[2,4,6,8,10]]), np.array([[ 2.5,  5. ], [ 5.,  10. ]]))
            ]

def test_covariance_using_transpose_method(covariance_test_data):
    np.testing.assert_almost_equal(get_covariance_transpose(covariance_test_data[0][0]),
                                                            covariance_test_data[0][1],
                                                            err_msg="Covariance matrices are not equal")

def test_covariance_using_cov_calculations(covariance_test_data):
    np.testing.assert_almost_equal(get_covariance_cov_calc(covariance_test_data[0][0]),
                                                            covariance_test_data[0][1],
                                                            err_msg="Covariance matrices are not equal")

def test_inverse():
    np.testing.assert_equal(get_inverse_2x2_matrix(np.array([[2,3],[1,4]])),
                                   np.array([[ 0.8, -0.6], [-0.2, 0.4]]))
