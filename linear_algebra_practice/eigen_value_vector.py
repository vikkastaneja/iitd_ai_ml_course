import numpy as np
import math

# This file contains code for calculating Eigen Values and Eigen Vectors for 2D arrays
# Eigen vector condition: AX=cX where A is transformation, X is a vector, c is a scalar
# In order to do it in python without using libraries in numpy:
# 1. (A-cI)X = 0 where X != 0
# 2. A-cI = 0
# 3. det(A-cI) = 0
# 4. For 2D matrix, expand det(A-cI) ==> c^2 - c * trace(A) + det(A)
#    where det(A) is simply A[0][0] * A[1][1] - A[1][0] * A[0][1],
#    and trace(A) = A[0][0] + A[1][1] --> sum of diagonal elements
# 5. Solve for c and that are eigen values using (-b + SQRT(d))/2a and (-b - SQRT(d))/2a
# 6. Solve the system of linear equations AX = cX for each 'c' and we get eigen vectors
# Calculate one possible eigen vector based on eigen value as follows:
# Let A (2D matrix) and X (Vector) are as follows:
"""
A = [a1 a2]
    [b1 b2]

X = [X]
    [Y]

To solve:
AX=cX

[a1 a2]  [X]  =  c[X]
[b1 b2]  [Y]  =  c[Y]

a1*X + a2*Y = c*X
b1*X + b2*Y = c*Y

Solve any equation:
(a1-c)*X = -a2*Y
X = (-a2/(a1-c))*Y

Let d = (-a2/(a1-c))

One possible solution is: [d, 1], which is also an Eigen Vector
Optionally, normalize it by dividing each value in vector by vector's value:
So, the normalized vector becomes:
[d/sqrt(d^2+1^2), 1/sqrt(d^2+1^2)], which is also an Eigen Vector
"""

# The following method finds out coefficients of c^2, c and constant terms of quadratic equation in point 4 above
def get_coefficients_2d_matrix(A):
    trace = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return [1, -trace, det]

# The following method solve for 'c' and find eigen values
def get_eigen_values_2d_matrix(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = b*b - 4*a*c
    s1 =(-b + math.sqrt(d)) / 2 * a
    s2 =(-b - math.sqrt(d)) / 2 * a
    return np.array([s1, s2])

# Calculate one possible eigen vector
def get_one_eigen_vector(A, c, array_index):
    # simplify based on the eigen value
    unit_array = []
    if array_index == 0:
        unit_array = [1, 0]
    elif array_index == 1:
        unit_array = [0, 1]

    arr = A - np.multiply(unit_array, c)

    tempx = -arr[1]
    tempy = arr[0]

    if tempy != 0:
        tempx = tempx/tempy
        tempy = 1

    div = np.sqrt(np.square(tempx) + np.square(tempy))
    A_final = np.array([tempx/div, tempy/div])
    print (A_final)
    return A_final

# The following function finds the eigen vectors from the A and c above
def get_eigen_vector(A, c):
    return get_one_eigen_vector(A[0], c, 0)

def get_eigens(A):
    coeffs = get_coefficients_2d_matrix(A)
    print(f"Coefficients of quadratic equation: {coeffs}")

    evalues = get_eigen_values_2d_matrix(coeffs)

    print(evalues)
    eigens = []
    for evalue in evalues:
        ev = get_eigen_vector(A=A, c=evalue)
        if ev is not None:
            eigens.append(np.array([evalue, ev], dtype=object))
        else:
            print(f"No unique eigen vector for {evalue}")

    for v in eigens:
        print(v)

    return eigens
