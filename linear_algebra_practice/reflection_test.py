import sys

sys.path.append('./')
sys.path.append('linear_algebra_practice')
from reflection import find_reflection

import numpy as np
import unittest
class TryReflection(unittest.TestCase):
    def test_reflection_on_x_axis(self):
        actual = find_reflection(np.array([2,3]), np.array([1,0]))
        expected = np.array([2., 0.])
        np.testing.assert_array_equal(actual, expected, "Results are different")
