import numpy as np

# Funções base de Diferenças Divididas de Newton
def diferencas_divididas_newton(x_nos, y_nos):
    n = len(x_nos)
    tabela = np.zeros((n, n))
    tabela[:, 0] = y_nos
    for j in range(1, n):
        for i in range(n - j):
            tabela[i, j] = (tabela[i + 1, j - 1] - tabela[i, j - 1]) / (x_nos[i + j] - x_nos[i])

    # Retorna a tabela para imprimir a matriz
    return tabela

def avaliar_polinomio_newton(x_nos, coef, x_alvo):
    resultado = coef[0]
    produto = 1.0
    for i in range(1, len(coef)):
        produto *= (x_alvo - x_nos[i - 1])
        resultado += coef[i] * produto
    return resultado

def construir_equacao_newton(x_nos, coef):
    termos = []
    for i in range(len(coef)):
        # Ignora apenas coeficientes exatamente nulos (os de Ex3 e Ex4 são pequenos, mas não zero)
        if coef[i] == 0:
            continue
            
        termo_atual = f"{coef[i]:.4g}"
        
        produto_raizes = ""
        for j in range(i):
            produto_raizes += f"(x - {x_nos[j]:.1f})"
            
        termos.append(termo_atual + produto_raizes)
        
    # Junta tudo e ajusta sinais negativos
    equacao = " + ".join(termos).replace("+ -", "- ")
    return f"p(x) = {equacao}"

# Ex 2
if __name__ == '__main__':
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([14.5, 19.5, 30.5, 53.5, 94.5, 159.5])

    tabela = diferencas_divididas_newton(x, y)
    coef = tabela[0, :] # Os coeficientes são a primeira linha da tabela

    print("Tabela de Diferenças Divididas:")
    print(f"{'x':<4} | {'Ordem 0 (y)':<11} | {'Ordem 1':<7} | {'Ordem 2':<7} | {'Ordem 3':<7} | {'Ordem 4':<7} | {'Ordem 5':<7}")
    print("-" * 77)
    for i in range(len(x)):
        linha = f"{x[i]:<4.1f} | {tabela[i, 0]:<11.1f}"
        for j in range(1, len(x) - i):
            linha += f" | {tabela[i, j]:<7.4g}"
        print(linha)

    print("\nCoeficientes de Newton:")
    print(np.round(coef, 6))

    equacao_gerada = construir_equacao_newton(x, coef)
    print(equacao_gerada)

    # Estimativa de f(4.5)
    xp = 4.5
    f_est = avaliar_polinomio_newton(x, coef, xp)
    f_exato = 71.375

    print(f"\nf({xp}) estimado = {f_est:.3f}")
    print(f"f({xp}) exato    = {f_exato:.3f}")