import numpy as np

# Dados
anos = np.arange(1959, 2026)
co2 = np.array([
    315.98, 316.91, 317.64, 318.45, 318.99, 319.62, 320.04, 321.37, 322.18, 323.05,
    324.62, 325.68, 326.32, 327.46, 329.68, 330.19, 331.13, 332.03, 333.84, 335.41,
    336.84, 338.76, 340.12, 341.48, 343.15, 344.87, 346.35, 347.61, 349.31, 351.69,
    353.20, 354.45, 355.70, 356.54, 357.21, 358.96, 360.97, 362.74, 363.88, 366.84,
    368.54, 369.71, 371.32, 373.45, 375.98, 377.70, 379.98, 382.09, 384.02, 385.83,
    387.64, 390.10, 391.85, 394.06, 396.74, 398.81, 401.01, 404.41, 406.76, 408.72,
    411.65, 414.21, 416.41, 418.53, 421.08, 424.61, 427.35,
])

t = anos - 1959   # anos deslocados: t = 0 em 1959
y = co2

def minimos_quadrados(x, y, n):
    """Ajusta um polinomio de grau n por minimos quadrados.

    Monta a matriz X (colunas 1, x, x^2, ..., x^n), resolve as equacoes
    normais (X^T X) c = X^T y e devolve:
      c    - coeficientes do polinomio (do termo constante ao de grau n)
      S    - soma dos quadrados dos residuos
      cond - numero de condicao da matriz A = X^T X
    """
    x = np.array(x, dtype=float)          # float evita estouro de inteiro em x**n
    X = np.zeros((len(x), n + 1))
    for j in range(n + 1):
        X[:, j] = x**j                    # coluna j = x elevado a j
    A = X.T @ X                           # matriz das equacoes normais
    b = X.T @ y                           # lado direito das equacoes normais
    c = np.linalg.solve(A, b)             # resolve A c = b
    S = np.sum((y - X @ c)**2)            # soma dos quadrados dos residuos
    return c, S, np.linalg.cond(A)        # cond(A): sensibilidade do sistema a erros


def ajuste_exponencial_ppm(t, y, A0, k0, iteracoes=50):
    """Ajusta y = 280 + A*exp(k*t) minimizando o S em ppm (escala original).

    Usa o metodo de Gauss-Newton: a cada iteracao linearizar o modelo em torno
    de (A, k), resolver um problema de minimos quadrados para a correcao
    (dA, dk) e atualizar os parametros. Comeca em (A0, k0) e para quando a
    correcao fica desprezivel. Devolve (A, k).
    """
    A, k = A0, k0
    for _ in range(iteracoes):
        e = np.exp(k * t)
        r = y - 280 - A * e                       # residuos em ppm
        J = np.column_stack([e, A * t * e])       # derivadas do modelo em A e em k
        delta = np.linalg.solve(J.T @ J, J.T @ r) # correcao (dA, dk)
        A += delta[0]
        k += delta[1]
        if np.max(np.abs(delta)) < 1e-12:         # convergiu
            break
    return A, k

# P5.1
print("P5.1")
print("grau   cond(A) ano     S ano          cond(A) t       S t")
S_ano = {}
S_t = {}
for n in [1, 2, 3, 4]:
    _, S1, cond1 = minimos_quadrados(anos, y, n)
    _, S2, cond2 = minimos_quadrados(t, y, n)
    S_ano[n] = S1
    S_t[n] = S2
    print(f"{n}      {cond1:.3e}     {S1:.6f}    {cond2:.3e}     {S2:.6f}")

# P5.2
print("\nP5.2")
for n in [1, 2, 3, 4]:
    print(f"grau {n}: diferenca entre os S = {abs(S_ano[n] - S_t[n]):.2e}")

# PB.1
assert np.all(y - 280 > 0)                  # o log exige C - 280 > 0
z = np.log(y - 280)
c, S_log, _ = minimos_quadrados(t, z, 1)
lnA = c[0]                                  # coeficiente linear = ln A
k = c[1]                                    # coeficiente angular = k
A = np.exp(lnA)                             # volta a escala original
dobra = np.log(2) / k                       # tempo de dobra: e^(k T) = 2

print("\nPB.1")
print(f"ln A = {lnA:.6f}   A = {A:.4f}   k = {k:.6f}")
print(f"tempo de dobra do excesso de CO2 = {dobra:.2f} anos")

# PB.2
t450 = np.log(170 / A) / k
print("\nPB.2")
print(f"t = {t450:.2f}, ou seja, ano {1959 + t450:.2f}")

S_exp = np.sum((y - 280 - A * np.exp(k * t))**2)   # S em ppm do ajuste no log
_, S_parab, _ = minimos_quadrados(t, y, 2)         # S da parabola
print(f"S em ppm do exponencial = {S_exp:.4f}")
print(f"S da parabola           = {S_parab:.4f}")

# PB.3
A2, k2 = ajuste_exponencial_ppm(t, y, A, k)
S_ppm_min = np.sum((y - 280 - A2 * np.exp(k2 * t))**2)
ano450_ppm = 1959 + np.log(170 / A2) / k2

print("\nPB.3")
print(f"soma minimizada (escala log) = {S_log:.6f}")
print(f"ajuste no log:      A = {A:.2f}  k = {k:.5f}  S em ppm = {S_exp:.1f}")
print(f"minimo real em ppm: A = {A2:.2f}  k = {k2:.5f}  S em ppm = {S_ppm_min:.1f}")
print(f"ano dos 450 ppm: {1959 + t450:.2f} (ajuste no log), {ano450_ppm:.2f} (minimo em ppm)")