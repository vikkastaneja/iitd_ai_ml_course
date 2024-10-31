import numpy as np
import sys
import os

parent_dir = ".."
libraries = "common_library"
current_dir = os.path.dirname(__file__)
s = [current_dir, parent_dir,libraries, 'src', 'common_functions']
libraries_path = os.path.join(*s)
sys.path.append(libraries_path)

from enumerators import NumOfSolutions

# Find out if there are none, unique or many solutions to system of linear equations
# AX = b

def identify(A, b) -> NumOfSolutions:
    try:
        x = np.linalg.solve(A, b)
        return NumOfSolutions.ONE
    except np.linalg.LinAlgError as e:
        print (e)
        if str(e) == "Singular matrix":
            return NumOfSolutions.MANY
        else:
            return NumOfSolutions.ONE

# 2x + 3y = 5
# 4x + 6y = 10
print("Equation set 1:")
A = np.array([[2,3],[4,6]])
b = np.array([5, 10])
print(identify(A, b))

# x - y + z = 2
# 2x + y + z = 5
# 3x + 2y + 2z = 7
print("Equation set 2:")
A = np.array([[1, -1, 1], [2, 1, 1], [3, 2, 2]])
b = np.array([2, 5, 7])
print(identify(A, b))

# 3x + 2y = 8
# 6x + 4y = 16
# 9x + 6y = 24
print("Equation set 3:")
A = np.array([[3, 2], [6, 4], [9, 6]])
b = np.array([8, 16, 24])
print(identify(A, b))

# x + 2y + z = 4
# 2x - y + 3z = 1
# -x + y + 2z = 3
print("Equation set 4:")
A = np.array([[1, 2, 1], [2, -1, 3], [-1, 1, 2]])
b = np.array([4, 1, 3])
print(identify(A, b))

# x + y + z = 1
# 2x + 3y + 2z = 4
# 3x + 5y + 3z = 5
print("Equation set 5:")
A = np.array([[1, 1, 1], [2, 3, 2], [3, 5, 3]])
b = np.array([1, 4, 5])
print(identify(A, b))
