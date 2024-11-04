import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')

from principal_component_analysis import get_pca_array_2d_matrix as get_pca
import pytest
import numpy as np

@pytest.fixture
def testdata():
    return [(np.array([[2.5, 2.4],[0.5, 0.7],[2.2,2.9],[1.9,2.2],[3.1,3],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]]),
    ([[3.45911227],[0.85356176],[3.62333958],[2.9053525 ],[4.3069435 ],[3.54409119],[2.53203265],[1.48656992],[2.19309595],[1.40732153]]), 96.31813143486461)]

def test_ranks(testdata):
    for data in testdata:
        actual = get_pca(data[0].T)
        print(f"-------> {data[1]}, {type(data[1])}")
        print(f"=======> {actual[0]}, {type(actual[0])}")
        np.testing.assert_almost_equal(actual[0], np.array(data[1]), err_msg="Failed validating reduced array")
        assert actual[1] == data[2]