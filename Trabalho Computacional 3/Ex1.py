import numpy as np

# Ex1
x = np.array([1.2, 2.1, 3.0, 3.6])
y = np.array([0.7, 8.1, 27.7, 45.1])

# O sistema linear V * a = y forma-se com V sendo a matriz de Vandermonde
V = np.vander(x, increasing=True)
a = np.linalg.solve(V, y)

print("Coeficientes do polinômio (a0 + a1*x + a2*x^2 + a3*x^3):")
for i, c in enumerate(a):
    print(f"a{i} = {c:.6f}")

print(f"\np(x) = {a[0]:.6f} + ({a[1]:.6f})x + ({a[2]:.6f})x^2 + ({a[3]:.6f})x^3")