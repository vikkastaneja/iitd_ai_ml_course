import numpy as np

# Eigen vector condition: AX=cX where A is transformation, X is a vector, c is a scalar
# In order to do it in python without using libraries in numpy:
# 1. (A-cI)X = 0 where X != 0
# 2. A-cI = 0
# 3. det(A-cI) = 0
# 4. For 2D matrix, expand det(A-cI) ==> c^2 - c * trace(A) + det(A)
# 5. Solve for c and that are eigen values
# 6. Solve the system of linear equations (A-cI)X = 0 for each C and we get eigen vectors 


# The following method finds out coefficients of c^2, c and constant term of quadratic equation in 4
def get_coefficients_2d_matrix(A):
    trace = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return [1, -trace, det]

# The following method solve for 'c' and find eigen values
def get_eigen_values_2d_matrix(coeffs):
    return np.roots(coeffs)


# Calculate one possible eigen vector based on eigen value
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
