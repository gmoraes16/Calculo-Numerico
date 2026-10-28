import numpy as np
from Ex2P2 import minimos_quadrados_polinomial

# Dados do Ex 5

x = np.array([1.2, 3.0, 4.5, 5.6, 6.9, 8.1])
y = np.array([2.3, 3.3, 5.6, 15.4, 38.2, 59.1])

# Grau = 2
A2, b2, a2, erroE2 = minimos_quadrados_polinomial(x, y, 2)

print("Grau = 2")
print("Matriz dos Somatórios (A):\n", A2)
print("Vetor independente (b):\n", b2)
print(f"p2(x) = {a2[0]:.4f} + ({a2[1]:.4f})x + ({a2[2]:.4f})x^2")
print(f"Erro Quadrático Total (E) = {erroE2:.4f}\n")

# Aproximação de f(6.0) usando p2(x)
x_val = 6.0
p2_val = a2[0] + a2[1] * x_val + a2[2] * (x_val**2)
print(f"Aproximação de f(6.0): {p2_val:.4f}\n")