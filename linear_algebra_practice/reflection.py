# Reflection of a vector X on L is
# R = Ref(X) = c*Vec(V) where V is an existing vector and c is a real number
# P = Perpendicular vector = X - R
# P is orthogonal to V ==> dot product of P and V = 0
# (X-R).V = 0 => (X-cV).V = 0 => X.V - cV.V = 0 => c = (X.V)/(V.V) => a scalar
# R = cV = ((X.V)/(V.V))V
'''
To understand how P, X, V and R are related:

              ^
             /|
            / |
           /  |
          /   |  P
         /    |
    X   /     |
       /      |
      /       |
(0, 0)-------->------------->
        R             V
    R + P = X ==> P = X - R (which is a constant times V)

'''
import numpy as np

def scalar_factor(X, V) -> float:
    c = round(np.dot(X, V)/np.dot(V, V), 2)
    print(f"Scalar factor: {c}")
    return c

def find_reflection(X, V):
    R = scalar_factor(X, V) * V
    print(f"Reflection of {X} on {V} is {R}")
    return R
