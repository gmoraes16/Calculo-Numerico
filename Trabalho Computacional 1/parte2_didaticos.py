import numpy as np
import math
import pandas as pd

# Importa as funções de métodos numéricos do arquivo metodos.py
from metodos import bisseccao, newton, secante

# 2.1
def tabelar_sinais(f, a, b, n):
    """
    Avalia f em n pontos igualmente espaçados no intervalo [a, b] 
    e retorna uma lista de intervalos onde há mudança de sinal.
    """
    # np.linspace cria um vetor com 'n' pontos de 'a' até 'b'
    pontos_x = np.linspace(a, b, n)
    intervalos = []
    
    for i in range(n - 1):
        x_atual = pontos_x[i]
        x_prox = pontos_x[i+1]
        
        f_atual = f(x_atual)
        f_prox = f(x_prox)
        
        # Se a multiplicação for menor ou igual a zero, houve mudança de sinal (ou bateu no zero)
        if f_atual * f_prox <= 0:
            intervalos.append((x_atual, x_prox))
            
    return intervalos

def exercicio_2_1():
    print("=== 2.1 Isolamento ===")
    
    # a) Teste com a função f(x)
    f = lambda x: x**3 - 9*x + 3
    
    print("\n(a) Função f(x) = x^3 - 9x + 3 no intervalo [-5, 5]:")
    para_n = [21, 11, 6, 4]
    
    for n in para_n:
        intervs = tabelar_sinais(f, -5, 5, n)
        print(f"Malha n={n:3d} encontrou {len(intervs)} raízes. Intervalos: {intervs}")

    # -----------------------------------------------------
    # b) Teste com a função g(x)
    g = lambda x: (x - 1.05) * (x - 1.15) * (x - 3)
    
    print("\n(b) Função g(x) = (x - 1.05)(x - 1.15)(x - 3) no intervalo [0, 4]:")
    para_n = [9, 17, 41, 401]
    
    for n in para_n:
        intervs = tabelar_sinais(g, 0, 4, n)
        print(f"Malha n={n:3d} encontrou {len(intervs)} raízes. Intervalos: {intervs}")

# 2.2
def exercicio_2_2():
    print("\n=== 2.2 Previsão x Realidade na bissecção ===")

    f = lambda x: x**3 - 9*x + 3
    a = 0.0
    b = 1.0
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

    resultados = []

    for eps in epsilons:
        # Cálculo da previsão pela fórmula do PDF
        k_float = (math.log10(b - a) - math.log10(eps)) / math.log10(2)

        # Como k deve ser maior que o valor calculado, usamos o teto da função para arredondar para cima
        k_previsto = math.ceil(k_float)

        # Cálculo da realidade usando a função bisseccao
        raiz, historico = bisseccao(f, a, b, eps=eps) # Implementação anterior
        k_real = len(historico) # O número de iterações realizadas é o tamanho do histórico

        resultados.append({
            "eps (Critério de parada)": eps,
            "k_previsto": k_previsto,
            "k_real": k_real,
        })

    # Criação do DataFrame para exibir os resultados de forma tabular
    tabela_resultados = pd.DataFrame(resultados)
    print(tabela_resultados.to_string(index=False))

# 2.3
def exercicio_2_3():
    print("\n=== 2.3 Custo real: Avaliações de função ===")

    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*x**2 - 9
    eps = 1e-8
    
    # Exibe o critério de parada cumprindo a regra do PDF
    print(f"Critério de parada utilizado para todos os métodos: eps = {eps}")

    # Cálculo da bissecção
    raiz_b, hist_b = bisseccao(f, 0.0, 1.0, eps=eps)

    # Cálculo do método de Newton (pede um chute x0, será usado x0=0.5, que é o ponto médio do intervalo)
    raiz_n, hist_n = newton(f, df, 0.5, eps=eps)

    # Cálculo do método da secante (pede dois chutes x0 e x1, será usado x0=0.0 e x1=1.0, que são os limites do intervalo)
    raiz_s, hist_s = secante(f, 0.0, 1.0, eps=eps)

    dados = [
        { 
            "Método": "Bissecção",
            "Iterações": len(hist_b),
            "Avaliações de f": hist_b[-1]["chamadas_f"],
            "Avaliações de df": 0
        },
        {
            "Método": "Newton",
            "Iterações": len(hist_n),
            "Avaliações de f": hist_n[-1]["chamadas_f"],
            "Avaliações de df": hist_n[-1]["chamadas_df"]
        },
        { 
            "Método": "Secante",
            "Iterações": len(hist_s),
            "Avaliações de f": hist_s[-1]["chamadas_f"],
            "Avaliações de df": 0
        }
    ]

    # Criação do DataFrame para exibir os resultados de forma tabular
    tabela_2_3 = pd.DataFrame(dados)
    print("\n" + tabela_2_3.to_string(index=False))

# 2.4
def calcular_ordem_empirica(historico, raiz_exata):
    """Calcula a ordem empírica de convergência com base no histórico de erros"""
    dados_ordens = []
    
    for k in range(2, len(historico)):
        # Pega os três últimos valores de x do histórico que são necessárias para a fórmula
        x_k = historico[k]["x"]
        x_km1 = historico[k-1]["x"]
        x_km2 = historico[k-2]["x"]

        # Calcula os erros em relação à raiz informada no PDF
        erro_k = abs(x_k - raiz_exata)
        erro_km1 = abs(x_km1 - raiz_exata)
        erro_km2 = abs(x_km2 - raiz_exata)

        if erro_k == 0 or erro_km1 == 0 or erro_km2 == 0:
            break  # Evita divisão por zero, caso a raiz exata seja atingida
        if erro_km1 == erro_km2:
            break

        # Fórmula da ordem empírica de convergência
        ordem = math.log(erro_k / erro_km1) / math.log(erro_km1 / erro_km2)

        # Guarda os dados formatados para exibição
        dados_ordens.append({
            "k (Iteração)": historico[k]["k"],
            "Erro exato": f"{erro_k:.3e}",
            "p_k empírico": round(ordem, 5)
        })

    return dados_ordens

def exercicio_2_4():
    print("\n=== 2.4 Ordem Empírica de Convergência ===")

    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*x**2 - 9

    # Raiz exata que está no PDF
    xi = 0.337608955965837
    eps = 1e-8
    
    print(f"Critério de parada eps = {eps}. Resultados processados até atingir precisão.\n")

    # Executa os métodos de newton e da secante
    raiz_n, hist_n = newton(f, df, 0.5, eps=eps)
    raiz_s, hist_s = secante(f, 0.0, 1.0, eps=eps)

    # Processa os dados
    dados_newton = calcular_ordem_empirica(hist_n, xi)
    dados_secante = calcular_ordem_empirica(hist_s, xi)

    # Criação dos DataFrames para exibir os resultados de forma tabular
    print("Ordem empírica de convergência - Método de Newton (Teórico p = 2)")
    tabela_newton = pd.DataFrame(dados_newton)
    print(tabela_newton.to_string(index=False))

    print("\nOrdem empírica de convergência - Método da Secante (Teórico p ≈ 1.618)")
    tabela_secante = pd.DataFrame(dados_secante)
    print(tabela_secante.to_string(index=False))

# 2.5
def exercicio_2_5():
    print("\n=== 2.5 Os modos de falha de Newton ===")

    # a)
    print("\n[Caso A] f(x) = x^3 - 2x + 2 | Parada forçada: max_iter = 10")
    fa = lambda x: x**3 - 2*x + 2
    dfa = lambda x: 3*x**2 - 2

    # Roda por 10 iterações
    _, hist_a = newton(fa, dfa, x0=0.0, max_iter=10)
    for passo in hist_a:
        print(f"k={passo['k']}, x={passo['x']:.5f}")

    # -----------------------------------------------------
    # b)
    print("\n[Caso B] f(x) = arctan(x)")
    fb = lambda x: np.arctan(x)
    dfb = lambda x: 1/(1 + x**2)

    print("Tentativa com x0 = 2.0 (Parada forçada: max_iter = 5):")
    _, hist_b = newton(fb, dfb, x0=2.0, max_iter=5)
    for passo in hist_b:
        print(f"k={passo['k']}, x={passo['x']:.5f}")

    print(f"\nTentativa com x0 = 1.0 (Critério de parada: eps = 1e-8):")
    _, hist_b2 = newton(fb, dfb, x0=1.0, eps=1e-8)
    for passo in hist_b2:
        print(f"k={passo['k']}, x={passo['x']:.5f}")

    # -----------------------------------------------------
    # c)
    print("\n[Caso C] f(x) = x^3 - 9x + 3 | x0 = sqrt(3)")
    fc = lambda x: x**3 - 9*x + 3
    dfc = lambda x: 3*x**2 - 9

    try:
        _, hist_c = newton(fc, dfc, x0=math.sqrt(3), max_iter=1)
        explosao_infinito = hist_c[0]   
        print(f"k=1: x={explosao_infinito['x']:.5e}")
    except Exception as e:
        print(f"A implementação interceptou a falha com o erro: {e}")

# 2.6
def newton_modificado(f, df, x0, m, eps=1e-8, max_iter=100):
    """Método de Newton Modificado com fator multiplicidade 'm'."""
    historico = []
    x = x0
    
    for k in range(max_iter):
        fx = f(x)
        historico.append({"k": k, "x": x})
        
        if abs(fx) < eps:
            break
            
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"Derivada zero em x={x}")
            
        x_prox = x - m * (fx / dfx) 
        
        if abs(x_prox - x) < eps:
            historico.append({"k": k+1, "x": x_prox})
            x = x_prox
            break
            
        x = x_prox
        
    return x, historico

def exercicio_2_6():
    print("\n=== 2.6 Raiz múltipla ===")
    
    f = lambda x: (x - 2)**2 * (x + 1)
    df = lambda x: 3*x**2 - 6*x

    xi = 2.0 # Raiz exata do PDF
    x0 = 3.0 # Chute inicial
    
    def imprimir_tabela_erro(historico, nome_metodo):
        print(f"\n--- {nome_metodo} ---")
        erros = [abs(passo["x"] - xi) for passo in historico]
        dados = []
        
        for k in range(len(historico) - 1):
            e_k = erros[k]
            e_k_prox = erros[k+1]
            razao = (e_k_prox / e_k) if e_k != 0 else 0
            
            dados.append({
                "k (Iteração)": historico[k]["k"],
                "x_k": f"{historico[k]['x']:.6f}",
                "e_k": f"{e_k:.2e}",
                "e_{k+1} / e_k": f"{razao:.5f}"
            })
            
        dados.append({
            "k (Iteração)": historico[-1]["k"],
            "x_k": f"{historico[-1]['x']:.6f}",
            "e_k": f"{erros[-1]:.2e}",
            "e_{k+1} / e_k": "-"
        })
        
        print(pd.DataFrame(dados).to_string(index=False))

    print("Critério de parada para ambos: eps=1e-8 ou max_iter=10")

    # Executa o método de Newton tradicional
    _, hist_n = newton(f, df, x0, eps=1e-8, max_iter=10)
    imprimir_tabela_erro(hist_n, "Newton Tradicional (m=1)")

    # Executa o método de Newton modificado com m=2
    _, hist_mod = newton_modificado(f, df, x0, m=2, eps=1e-8, max_iter=10)
    imprimir_tabela_erro(hist_mod, "Newton Modificado (m=2)")

# 2.7
def exercicio_2_7():
    print("\n=== 2.7 Armadilha de Resíduo ===")

    f = lambda x: (x-1)**10
    raiz_exata = 1.0

    # 2.7a
    f_1_1 = f(1.1)
    f_1_3 = f(1.3)
    print("Avaliando a ordem de grandeza do resíduo (f(x)):")
    print(f"f(1.1) = {f_1_1:.2e}")
    print(f"f(1.3) = {f_1_3:.2e}")

    # Bissecção apenas com a parada enunciada no ponto 2.7b)
    def bisseccao2_7b(a, b, eps=1e-8):
        """Implementação proposital da bissecção contendo apenas critério de parada por resíduo."""
        k = 0
        while k < 200:
            x = (a + b) / 2.0
            fx = f(x)

            # Parada apenas pelo critério enunciado (|f(x)| < eps)
            if abs(fx) < eps:
                return x, k
            
            if f(a) * fx < 0:
                b = x
            else:
                a = x
            k += 1
        return (a + b) / 2.0, k
    
    # -----------------------------------------------------
    # Bissecção apenas com a parada enunciada no ponto 2.7c)
    def bisseccao2_7c(a, b, eps=1e-8):
        """Implementação proposital da bissecção contendo apenas critério de parada por passo."""
        k = 0
        x_ant = a
        while k < 200:
            x = (a + b) / 2.0

            # Parada apenas pelo critério enunciado (|x_k+1 - x_k| < eps)
            if abs(x - x_ant) < eps:
                return x, k
            
            if f(a) * f(x) < 0:
                b = x
            else:
                a = x
            x_ant = x
            k += 1
        return (a + b) / 2.0, k

    # Executa os métodos e compara
    raiz_res, k_res = bisseccao2_7b(0.0, 1.5, eps=1e-8)
    erro_res = abs(raiz_res - raiz_exata)
    
    raiz_pas, k_pas = bisseccao2_7c(0.0, 1.5, eps=1e-8)
    erro_pas = abs(raiz_pas - raiz_exata)

    print(f"\n--- Critério de parada em 2.7b: |f(x)| < 1e-8 ---")
    print(f"Raiz obtida: {raiz_res:.8f}")
    print(f"Erro real em x: {erro_res:.8f} | Iterações (k): {k_res}")

    print(f"\n--- Critério de parada em 2.7c: |x_k+1 - x_k| < 1e-8 ---")
    print(f"Raiz obtida: {raiz_pas:.8f}")
    print(f"Erro real em x: {erro_pas:.8f} | Iterações (k): {k_pas}")

# Execução Principal
if __name__ == "__main__":
    exercicio_2_1()
    exercicio_2_2()
    exercicio_2_3()
    exercicio_2_4()
    exercicio_2_5()
    exercicio_2_6()
    exercicio_2_7()