import numpy as np

# Formatação limpa com até 4 casas (sem pontos soltos)
np.set_printoptions(formatter={'float': lambda x: f"{x:.1f}" if np.round(x, 4).is_integer() else f"{np.round(x, 4):g}"})

def minimos_quadrados_polinomial(x, y, grau):
    n = len(x)
    # Inicializa a matriz A e o vetor b
    A = np.zeros((grau + 1, grau + 1))
    b = np.zeros(grau + 1)

    # Preenche com os somatórios conforme o slide
    for linha in range(grau + 1):
        for coluna in range(grau + 1):
            A[linha, coluna] = np.sum(x ** (linha + coluna))
        b[linha] = np.sum(y * (x ** linha))

    # Resolve o sistema linear A*a = b
    a = np.linalg.solve(A, b)

    # Calcula o polinômio ajustado p(X) para cada ponto x
    p_x = sum(a[i] * (x ** i) for i in range(grau + 1))

    # Calcula o erro quadrático
    erroE = np.sum((p_x - y) ** 2)

    return A, b, a, erroE

if __name__ == "__main__":
    # Dados do Ex 2
    x = np.array([1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8])
    y = np.array([5.5, 7.3, 14.1, 26.3, 48.2, 80.1, 111.6])

    A2, b2, a2, erroE2 = minimos_quadrados_polinomial(x, y, 2)
    print("a): Grau = 2")
    print("Matriz dos Somatórios (A):\n", A2)
    print("Vetor independente (b):\n", b2)
    print(f"p2(x) = {a2[0]:.4f} + ({a2[1]:.4f})x + ({a2[2]:.4f})x^2")
    print(f"Erro Quadrático Total (E) = {erroE2:.4f}\n")

    A3, b3, a3, erroE3 = minimos_quadrados_polinomial(x, y, 3)
    print("b): Grau = 3")
    print("Matriz dos Somatórios (A):\n", A3)
    print("Vetor independente (b):\n", b3)
    print(f"p3(x) = {a3[0]:.4f} + ({a3[1]:.4f})x + ({a3[2]:.4f})x^2 + ({a3[3]:.4f})x^3")
    print(f"Erro Quadrático Total (E) = {erroE3:.4f}\n")