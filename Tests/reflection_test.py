import sys
import os
import numpy as np

parent_dir = ".."
practice = "Practice"
current_dir = os.path.dirname(__file__)
final_name = os.path.join(current_dir, parent_dir)
practice_path = os.path.join(final_name, practice)
sys.path.append(practice_path)

from reflection import find_reflection, scalar_factor
import unittest
class TryReflection(unittest.TestCase):
    def test_reflection_on_x_axis(self):
        actual = find_reflection(np.array([2,3]), np.array([1,0]))
        expected = np.array([2., 0.])
        np.testing.assert_array_equal(actual, expected, "Results are different")
