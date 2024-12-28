import numpy as np
from scipy.optimize import linprog
from scipy.linalg import lu, qr, svd

def initialize_constraints():
    A = []
    b = []

    # Constraint baris
    for i in range(9):
        for num in range(1, 10):
            row_constraint = [0] * 729
            for j in range(9):
                row_constraint[i * 81 + j * 9 + (num - 1)] = 1
            A.append(row_constraint)
            b.append(1)

    # Constraint kolom
    for j in range(9):
        for num in range(1, 10):
            col_constraint = [0] * 729
            for i in range(9):
                col_constraint[i * 81 + j * 9 + (num - 1)] = 1
            A.append(col_constraint)
            b.append(1)

    # Constraint subgrid
    for box_row in range(3):
        for box_col in range(3):
            for num in range(1, 10):
                subgrid_constraint = [0] * 729
                for i in range(3):
                    for j in range(3):
                        r = box_row * 3 + i
                        c = box_col * 3 + j
                        subgrid_constraint[r * 81 + c * 9 + (num - 1)] = 1
                A.append(subgrid_constraint)
                b.append(1)

    # Constraint sel
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
        return R  # Representasikan constraint dengan R
    elif method == "lu":
        P, L, U = lu(A)
        return U  # Representasikan constraint dengan U
    elif method == "svd":
        U, S, Vt = svd(A, full_matrices=False)
        return np.diag(S) @ Vt
    else:
        raise ValueError("Metode dekomposisi tidak valid. Gunakan 'qr', 'lu', atau 'svd'.")

def solve_sudoku_with_decomposed_constraints(matrix, method="qr"):
    # Inisialisasi constraint
    A, b = initialize_constraints()

    # Dekomposisi untuk validasi atau preprocessing
    print(f"Using {method.upper()} Decomposition for validation...")
    A_decomposed = decompose_constraints(A, method=method)

    if method == "qr":
        Q, R = qr(A, mode='economic')
        b_decomposed = Q.T @ b  # Sesuaikan b
    elif method == "lu":
        P, L, U = lu(A)
        b_decomposed = np.linalg.inv(L) @ (P.T @ b)  # Sesuaikan b
    elif method == "svd":
        U, S, Vt = svd(A, full_matrices=False)
        A_decomposed = np.diag(S) @ Vt
        b_decomposed = U.T @ b  # Sesuaikan b
    else:
        raise ValueError("Metode dekomposisi tidak valid!")

    # Cek apakah dekomposisi sesuai
    if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_decomposed):
        raise ValueError("Dekomposisi menyebabkan kehilangan informasi pada constraint!")

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
        raise ValueError("Tidak dapat menyelesaikan Sudoku dengan constraint yang diberikan.")

def solve_sudoku(matrix):
    A, b = initialize_constraints()
    c = [0] * 729  # Fungsi tujuan tidak relevan untuk constraint satisfaction

    # Tambahkan constraint untuk angka yang sudah terisi
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

    # Solve menggunakan linprog
    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

    if res.success:
        solution = np.array(res.x).reshape((9, 9, 9))
        solved_sudoku = np.argmax(solution, axis=2) + 1
        return solved_sudoku
    else:
        raise ValueError("Tidak dapat menyelesaikan Sudoku.")

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
        [0, 0, 2, 0, 0, 7, 3, 0, 5],
        [8, 0, 0, 0, 3, 0, 0, 0, 6],
        [0, 0, 0, 6, 0, 0, 0, 9, 0],
        [7, 0, 3, 0, 0, 0, 0, 0, 2],
        [0, 9, 1, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 1, 3, 0, 0, 0],
        [0, 0, 6, 7, 0, 0, 9, 1, 0],
        [3, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 8, 0, 0, 0]
    ])

    print("Sudoku Puzzle:")
    print_sudoku(sudoku)

    print("Sudoku Solved by QR:")
    print_sudoku(solve_sudoku_with_decomposed_constraints(sudoku, method="qr"))

    print("Sudoku Solved by LU:")
    print_sudoku(solve_sudoku_with_decomposed_constraints(sudoku, method="lu"))

    print("Sudoku Solved by SVD:")
    print_sudoku(solve_sudoku_with_decomposed_constraints(sudoku, method="svd"))

    print("Sudoku Solved Real:")
    print_sudoku(solve_sudoku(sudoku))