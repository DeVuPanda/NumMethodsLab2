import numpy as np

A = np.array([[3, 2, -1, 0],
              [2, 4, 0, -1],
              [-1, 0, 5, 2],
              [0, -1, 2, 6]], dtype=float)

n = A.shape[0]

S = np.zeros((n, n), dtype=float)
D = np.zeros((n, n), dtype=float)

for i in range(n):
    sum_s2d = sum(S[p, i] ** 2 * D[p, p] for p in range(i))
    D[i, i] = np.sign(A[i, i] - sum_s2d)
    S[i, i] = np.sqrt(abs(A[i, i] - sum_s2d))

    for j in range(i + 1, n):
        sum_spd = sum(S[p, i] * D[p, p] * S[p, j] for p in range(i))
        S[i, j] = (A[i, j] - sum_spd) / (D[i, i] * S[i, i])

det_A = np.prod(np.diag(D)) * np.prod(np.diag(S)) ** 2

print("Матриця S:")
print(S)
print("Матриця D:")
print(D)
print(f"Детермінант матриці A: {det_A}")
