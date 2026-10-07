import numpy as np
from Ex2 import diferencas_divididas_newton, avaliar_polinomio_newton, construir_equacao_newton

# Ex 4
T = np.array([605, 685, 725, 765, 825, 855, 875], dtype=float)
C = np.array([0.622, 0.655, 0.668, 0.679, 0.730, 0.907, 1.336])

print(f"cond(V) = {np.linalg.cond(np.vander(T, increasing=True)):.2e}\n")

# Pontos de teste e valores conhecidos
T_teste = [645.0, 795.0, 845.0]
C_conhecido = [0.639, 0.694, 0.812]

# Calcula a tabela e extrai os coeficientes
tabela = diferencas_divididas_newton(T, C)
coef = tabela[0, :]

print("Coeficientes de Newton calculados:")
for k, c in enumerate(coef):
    print(f"c{k} = {c:.6e}")

print("\nEquação Final do Polinômio:")
# Gera a equação e substitui 'x' por 'T' e 'p(x)' por 'C(T)'
equacao = construir_equacao_newton(T, coef)
equacao = equacao.replace("p(x)", "C(T)").replace("x", "T")
print(equacao)

print("\nEstimativas:")
print(f"{'T':<5} | {'C estimado':<11} | {'C conhecido':<11} | {'Erro absoluto':<13}")
print("-" * 47)
for t, ck in zip(T_teste, C_conhecido):
    ce = avaliar_polinomio_newton(T, coef, t)
    ea = abs(ce - ck)
    print(f"{t:<5.0f} | {ce:<11.6f} | {ck:<11.6f} | {ea:<13.6f}")