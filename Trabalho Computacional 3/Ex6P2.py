import numpy as np
from Ex2P2 import minimos_quadrados_polinomial

# Dados do Ex 6
x = np.array([1.0, 2.3, 3.8, 4.9, 6.1, 7.2])
y = np.array([0.1, 1.5, 7.4, 15.2, 26.2, 27.9])

# Grau = 2
A2, b2, a2, erroE2 = minimos_quadrados_polinomial(x, y, 2)

print("Grau = 2")
print("Matriz dos Somatórios (A):\n", A2)
print("Vetor independente (b):\n", b2)
print(f"p2(x) = {a2[0]:.4f} + ({a2[1]:.4f})x + ({a2[2]:.4f})x^2")
print(f"Erro Quadrático Total (E) = {erroE2:.4f}\n")