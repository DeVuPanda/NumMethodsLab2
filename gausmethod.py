import numpy as np  

def gauss_elimination(A, b):
    n = len(b)  
    determinant = 1  
    swap_count = 0  

    print("Початкова матриця A та вектор b:")
    print(A)
    print(b)
    print("-" * 40)

    for i in range(n):  
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i: 
            A[[i, max_row]] = A[[max_row, i]] 
            b[i], b[max_row] = b[max_row], b[i]  
            swap_count += 1  

        sub = A[i][i]
        determinant *= sub

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

        print(f"Ітерація {i + 1}:")
        print("Матриця A:")
        print(A)
        print("Вектор b:")
        print(b)
        print("-" * 40)

    determinant *= (-1) ** swap_count
    print(f"Визначник матриці: {determinant}")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]

    return x, determinant


def find_inverse(A):
    n = A.shape[0]
    A = np.copy(A)
    E = np.eye(n)
    A_inv = np.copy(E)

    print("Початкова матриця A та одинична матриця E:")
    print(A)
    print(E)
    print("-" * 40)

    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            A_inv[[i, max_row]] = A_inv[[max_row, i]]

        sub = A[i][i]
        A[i] = A[i] / sub
        A_inv[i] = A_inv[i] / sub

        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            A_inv[j] = A_inv[j] - factor * A_inv[i]

        print(f"Ітерація {i + 1} (прямий хід):")
        print("Матриця A:")
        print(A)
        print("Обернена матриця (на даний момент):")
        print(A_inv)
        print("-" * 40)

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            A_inv[j] = A_inv[j] - factor * A_inv[i]

        print(f"Ітерація {n - i} (зворотній хід):")
        print("Матриця A:")
        print(A)
        print("Обернена матриця (на даний момент):")
        print(A_inv)
        print("-" * 40)

    return A_inv



A = np.array([[3, 2, -1, 0],
              [2, 4, 0, -1],
              [-1, 0, 5, 2],
              [0, -1, 2, 6]], dtype=float)

b = np.array([5, 4, 1, 7], dtype=float)


solution, determinant = gauss_elimination(np.copy(A), b)
print("Корені системи:", solution)


A_inv = find_inverse(A)
print("Обернена матриця A_inv:")
print(A_inv)
