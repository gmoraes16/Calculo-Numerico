import numpy as np
from Ex2P2 import minimos_quadrados_polinomial

# Dados do Ex 4
x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
# Calcula f(x) = sin(pi * x) para cada valor de x e limpa resíduos
y = np.round(np.sin(np.pi * x), 10) + 0.0

print("Pontos calculados:")
print("x:", x)
print("f(x):", y, "\n")

# Grau = 3
A3, b3, a3, erroE3 = minimos_quadrados_polinomial(x, y, 3)
print("Grau = 3")
print("Matriz dos Somatórios (A):\n", A3)
print("Vetor independente (b):\n", b3)
print(f"p3(x) = {a3[0]:.4f} + ({a3[1]:.4f})x + ({a3[2]:.4f})x^2 + ({a3[3]:.4f})x^3")
print(f"Erro Quadrático Total (E) = {erroE3:.4f}\n")