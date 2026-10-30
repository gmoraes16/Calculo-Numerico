import numpy as np
from Ex2P2 import minimos_quadrados_polinomial

# Dados do Ex 3
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([7.0, 11.0, 17.0, 25.0])

# Grau = 2
A2, b2, a2, erroE2 = minimos_quadrados_polinomial(x, y, 2)
print("Grau = 2")
print("Matriz dos Somatórios (A):\n", A2)
print("Vetor independente (b):\n", b2)
print(f"p2(x) = {a2[0]:.4f} + ({a2[1]:.4f})x + ({a2[2]:.4f})x^2")
print(f"Erro Quadrático Total (E) = {erroE2:.4f}\n")

# Grau = 3
A3, b3, a3, erroE3 = minimos_quadrados_polinomial(x, y, 3)
a3_3 = round(a3[3], 10) + 0.0
print("Grau = 3")
print("Matriz dos Somatórios (A):\n", A3)
print("Vetor independente (b):\n", b3)
print(f"p3(x) = {a3[0]:.4f} + ({a3[1]:.4f})x + ({a3[2]:.4f})x^2 + ({a3[3]:.4f})x^3")
print(f"Erro Quadrático Total (E) = {erroE3:.4f}\n")

# Calculo dos polinômios nos 4 valores de x para comparação
p2_x = a2[0] + a2[1]*x + a2[2]*x**2
p3_x = a3[0] + a3[1]*x + a3[2]*x**2 + a3[3]*x**3

print("Comparação dos 4 valores de x:")
print("x:", x)
print("f(x):", y)
print("p2(x):", p2_x)
print("p3(x):", p3_x)
