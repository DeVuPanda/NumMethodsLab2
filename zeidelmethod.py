import numpy as np

def zeidel(A, b, epsilon):
    n = len(b)
    x = np.zeros_like(b)
    x_prev = np.zeros_like(b)
    condition_number = np.inf
    iteration = 0

    while condition_number > epsilon:
        iteration += 1
        print(f"\nІтерація {iteration}")
        x_prev = np.copy(x)

        for i in range(n):
            sum = 0
            for j in range(n):
                if j != i:
                    sum += A[i][j] * x[j]

            x[i] = (b[i] - sum) / A[i][i]

        print(f"x = {x}")

        condition_number = np.max(np.abs(x - x_prev))

        comparison_sign = ">=" if condition_number > epsilon else "<="

        print(f"||x{iteration} - x{iteration-1}|| = {condition_number}")
        print(f"Перевірка: {condition_number} {comparison_sign} {epsilon}")

    return x, iteration, condition_number

A = np.array([[3, 2, -1, 0],
              [2, 4, 0, -1],
              [-1, 0, 5, 2],
              [0, -1, 2, 6]], dtype=float)

b = np.array([5, 4, 1, 7], dtype=float)

epsilon = float(input("Введіть бажану точність (epsilon): "))

x, iterations, condition_number = zeidel(A, b, epsilon)

print(f"\nКорені системи: {x}")
