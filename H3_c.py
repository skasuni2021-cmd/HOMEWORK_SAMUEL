"""
hw3c.py
MAE 3403 – HW 3

Checks if matrix is symmetric and positive definite.
If yes → solves using Cholesky.
If not → solves using Doolittle LU factorization.

Only uses: math, copy
"""

import math
import copy


# ---------------------------------------------------------
# Check Symmetry
# ---------------------------------------------------------
def is_symmetric(A):

    n = len(A)

    for i in range(n):
        for j in range(n):

            if abs(A[i][j] - A[j][i]) > 1e-10:
                return False

    return True


# ---------------------------------------------------------
# Check Positive Definite using leading principal minors
# ---------------------------------------------------------
def is_positive_definite(A):

    n = len(A)

    for k in range(1, n+1):

        sub = [row[:k] for row in A[:k]]

        if determinant(sub) <= 0:
            return False

    return True


# ---------------------------------------------------------
# Determinant (recursive)
# ---------------------------------------------------------
def determinant(A):

    n = len(A)

    if n == 1:
        return A[0][0]

    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    det = 0

    for c in range(n):

        minor = [row[:c] + row[c+1:] for row in A[1:]]

        det += ((-1)**c) * A[0][c] * determinant(minor)

    return det


# ---------------------------------------------------------
# Cholesky Decomposition
# ---------------------------------------------------------
def cholesky(A):

    n = len(A)

    L = [[0.0]*n for _ in range(n)]

    for i in range(n):

        for j in range(i+1):

            sum_val = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:

                L[i][j] = math.sqrt(A[i][i] - sum_val)

            else:

                L[i][j] = (A[i][j] - sum_val) / L[j][j]

    return L


# ---------------------------------------------------------
# Forward Substitution
# ---------------------------------------------------------
def forward_substitution(L, b):

    n = len(b)

    y = [0]*n

    for i in range(n):

        y[i] = (b[i] - sum(L[i][j]*y[j] for j in range(i))) / L[i][i]

    return y


# ---------------------------------------------------------
# Backward Substitution
# ---------------------------------------------------------
def backward_substitution(U, y):

    n = len(y)

    x = [0]*n

    for i in reversed(range(n)):

        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]

    return x


# ---------------------------------------------------------
# Doolittle LU Decomposition
# ---------------------------------------------------------
def doolittle(A):

    n = len(A)

    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for j in range(n):

        for i in range(j+1):

            sum_val = sum(U[k][j]*L[i][k] for k in range(i))

            U[i][j] = A[i][j] - sum_val

        for i in range(j, n):

            sum_val = sum(U[k][j]*L[i][k] for k in range(j))

            L[i][j] = (A[i][j] - sum_val) / U[j][j]

    return L, U


# ---------------------------------------------------------
# Solve System
# ---------------------------------------------------------
def solve_system(A, b):

    if is_symmetric(A) and is_positive_definite(A):

        print("\nMatrix is symmetric and positive definite.")
        print("Using Cholesky Method.")

        L = cholesky(A)

        y = forward_substitution(L, b)

        LT = list(map(list, zip(*L)))

        x = backward_substitution(LT, y)

        method = "Cholesky"

    else:

        print("\nMatrix is NOT symmetric positive definite.")
        print("Using Doolittle LU Method.")

        L, U = doolittle(A)

        y = forward_substitution(L, b)

        x = backward_substitution(U, y)

        method = "Doolittle LU"

    return x, method


# ---------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------
def print_solution(x):

    print("\nSolution Vector:")

    for i, val in enumerate(x):

        print(f"x{i+1} = {val:.6f}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("Matrix Solver (Cholesky / Doolittle)")

    n = int(input("Enter size of matrix (n): "))

    A = []

    print("Enter matrix A row by row (space separated):")

    for i in range(n):

        row = list(map(float, input().split()))

        A.append(row)

    print("Enter vector b:")

    b = list(map(float, input().split()))

    x, method = solve_system(A, b)

    print_solution(x)

    print(f"\nMethod Used: {method}")


if __name__ == "__main__":
    main()