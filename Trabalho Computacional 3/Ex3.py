import numpy as np
from Ex2 import diferencas_divididas_newton, avaliar_polinomio_newton, construir_equacao_newton

# Ex 3
T = np.array([0, 10, 20, 30, 40, 60, 80, 100], dtype=float)
P = np.array([0.0061, 0.0123, 0.0234, 0.0424, 0.0738, 0.1992, 0.4736, 1.0133])

print(f"cond(V) = {np.linalg.cond(np.vander(T, increasing=True)):.2e}\n")

# Pontos de teste e valores conhecidos
T_teste = [5.0, 45.0, 95.0]
P_conhecido = [0.008721, 0.095848, 0.84528]

# Calcula a tabela inteira e extrai os coeficientes
tabela = diferencas_divididas_newton(T, P)
coef = tabela[0, :]

print("Coeficientes de Newton calculados:")
for k, c in enumerate(coef):
    print(f"c{k} = {c:.6e}")

print("\nEquação Final do Polinômio:")
# Gera a equação e substitui 'x' por 'T' e 'p(x)' por 'P(T)'
equacao = construir_equacao_newton(T, coef)
equacao = equacao.replace("p(x)", "P(T)").replace("x", "T")
print(equacao)

print("\nEstimativas:")
print(f"{'T':<5} | {'P estimado':<12} | {'P conhecido':<12} | {'Erro absoluto':<13}")
print("-" * 48)
for t, pk in zip(T_teste, P_conhecido):
    pe = avaliar_polinomio_newton(T, coef, t)
    ea = abs(pe - pk)
    print(f"{t:<5.0f} | {pe:<12.6f} | {pk:<12.6f} | {ea:<13.6f}")