# Name : Muhammad Fithra Rizki
# NIM : 13523049

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import lu, qr, svd

def initialize_constraints():
    A = []
    b = []

    # Row Constraint 
    for i in range(9):
        for num in range(1, 10):
            row_constraint = [0] * 729
            for j in range(9):
                row_constraint[i * 81 + j * 9 + (num - 1)] = 1
            A.append(row_constraint)
            b.append(1)

    # Column Constraint
    for j in range(9):
        for num in range(1, 10):
            col_constraint = [0] * 729
            for i in range(9):
                col_constraint[i * 81 + j * 9 + (num - 1)] = 1
            A.append(col_constraint)
            b.append(1)

    # Submatrix Constraint
    for box_row in range(3):
        for box_col in range(3):
            for num in range(1, 10):
                submatrix_constraint = [0] * 729
                for i in range(3):
                    for j in range(3):
                        r = box_row * 3 + i
                        c = box_col * 3 + j
                        submatrix_constraint[r * 81 + c * 9 + (num - 1)] = 1
                A.append(submatrix_constraint)
                b.append(1)

    # Cell Constraint
    for i in range(9):
        for j in range(9):
            cell_constraint = [0] * 729
            for num in range(1, 10):
                cell_constraint[i * 81 + j * 9 + (num - 1)] = 1
            A.append(cell_constraint)
            b.append(1)
    return np.array(A), np.array(b)


def decompose_constraints(A, method):
    if method == "qr":
        Q, R = qr(A, mode='economic')
        return R  # Constraint representation with R
    elif method == "lu":
        P, L, U = lu(A)
        return U  # Constraint representation with U
    elif method == "svd":
        U, S, Vt = svd(A, full_matrices=False) # Constraint representation with S and Vt
        return np.diag(S) @ Vt
    else:
        print("Decomposition method invalid. Use 'qr', 'lu', or 'svd'.")

def solve_sudoku(matrix, method="qr"):
    A, b = initialize_constraints()

    # Validating or preprocessing with decomposition
    print(f"Using {method.upper()} Decomposition for validation...")
    A_decomposed = decompose_constraints(A, method=method)
    if method == "qr":
        Q, R = qr(A, mode='economic')
        b_decomposed = Q.T @ b  # b value adjustment for QR
    elif method == "lu":
        P, L, U = lu(A)
        b_decomposed = np.linalg.inv(L) @ (P.T @ b)  # b value adjustment for LU
    elif method == "svd":
        U, S, Vt = svd(A, full_matrices=False)
        A_decomposed = np.diag(S) @ Vt
        b_decomposed = U.T @ b  # b value adjustment for SVD
    else:
        print("Decomposition method invalid!")

    # Check if the decomposition valid
    if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_decomposed):
        print("Decomposition causes loss of information on constraints!")
    bounds = []
    for i in range(9):
        for j in range(9):
            if matrix[i, j] != 0:
                num = matrix[i, j]
                for k in range(1, 10):
                    if k == num:
                        bounds.append((1, 1))
                    else:
                        bounds.append((0, 0))
            else:
                bounds.extend([(0, 1)] * 9)
    c = [0] * 729
    res = linprog(c, A_eq=A_decomposed, b_eq=b_decomposed, bounds=bounds, method="highs")

    if res.success:
        solution = np.array(res.x).reshape((9, 9, 9))
        solved_sudoku = np.argmax(solution, axis=2) + 1
        return solved_sudoku
    else:
        print("Sudoku can't be solved with the given constraints.")

def print_sudoku(matrix):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(matrix[i, j] if matrix[i, j] != 0 else ".", end=" ")
        print()


if __name__ == "__main__":
    sudoku = np.array([
        [0, 0, 0, 0, 9, 6, 0, 0, 0],
        [0, 4, 0, 2, 0, 8, 0, 6, 0],
        [0, 0, 0, 7, 5, 0, 8, 0, 2],
        [3, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 9, 0, 0, 5, 6, 3, 0],
        [0, 0, 0, 5, 3, 0, 0, 2, 0],
        [2, 5, 0, 0, 4, 0, 0, 1, 0],
        [6, 1, 0, 0, 0, 0, 4, 0, 5]
    ])

    print("Sudoku Puzzle:")
    print_sudoku(sudoku)

    print("Sudoku Solved by QR:")
    print_sudoku(solve_sudoku(sudoku, method="qr"))

    print("Sudoku Solved by LU:")
    print_sudoku(solve_sudoku(sudoku, method="lu"))

    print("Sudoku Solved by SVD:")
    print_sudoku(solve_sudoku(sudoku, method="svd"))



    #     EXTREME
        # [0, 0, 0, 0, 9, 6, 0, 0, 0],
        # [0, 4, 0, 2, 0, 8, 0, 6, 0],
        # [0, 0, 0, 7, 5, 0, 8, 0, 2],
        # [3, 0, 0, 0, 6, 0, 0, 0, 0],
        # [0, 0, 0, 3, 0, 0, 0, 0, 0],
        # [0, 0, 9, 0, 0, 5, 6, 3, 0],
        # [0, 0, 0, 5, 3, 0, 0, 2, 0],
        # [2, 5, 0, 0, 4, 0, 0, 1, 0],
        # [6, 1, 0, 0, 0, 0, 4, 0, 5]

        # EASY
        # [0, 0, 0, 0, 7, 4, 8, 0, 0],
        # [0, 0, 0, 0, 0, 8, 7, 2, 6],
        # [8, 0, 0, 0, 0, 0, 3, 0, 5],
        # [1, 0, 2, 0, 8, 0, 4, 0, 9],
        # [0, 0, 8, 4, 3, 0, 1, 6, 2],
        # [7, 0, 0, 0, 1, 2, 5, 0, 0],
        # [4, 0, 0, 8, 6, 5, 0, 0, 3],
        # [9, 0, 5, 0, 0, 3, 0, 0, 0],
        # [0, 8, 3, 0, 9, 0, 2, 5, 0]

        # MASTER
        # [0, 0, 0, 0, 4, 3, 0, 0, 2],
        # [0, 5, 0, 0, 0, 0, 0, 0, 0],
        # [7, 0, 4, 6, 0, 0, 1, 0, 0],
        # [0, 8, 0, 9, 0, 0, 0, 0, 0],
        # [0, 6, 0, 0, 0, 0, 5, 0, 0],
        # [9, 0, 5, 0, 2, 0, 0, 0, 1],
        # [8, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 6, 0, 0, 7, 0],
        # [1, 0, 9, 7, 0, 0, 4, 0, 0]