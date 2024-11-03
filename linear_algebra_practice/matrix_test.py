import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')

from matrix import find_rank_2d_matrix as find_rank
import pytest

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