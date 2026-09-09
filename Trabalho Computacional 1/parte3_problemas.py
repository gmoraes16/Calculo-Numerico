import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa os métodos numéricos implementados em metodos.py
from metodos import bisseccao, newton, secante

# Garante que a pasta usada para salvar os gráficos exista no projeto
os.makedirs("figuras", exist_ok=True)

# Problema A
def problema_A():
    print("=== Problema A - Reservatório esférico ===")
    
    R = 3.0
    volume_alvo = 40.0
    # O enunciado pede erro inferior a 1 mm (1e-3 m).
    eps_padrao = 1e-3
    max_iter_padrao = 100
    
    print(f"Critério de parada: eps = {eps_padrao}, max_iter = {max_iter_padrao}\n")
    
    # 1. Dedução da função f(h) cuja raiz resolve o problema
    # V(h) = pi * h^2 * (3R - h) / 3
    # f(h) = pi * h^2 * (3(3) - h) / 3 - 40 = 0
    def volume_esfera(h):
        return math.pi * (h**2) * (3 * R - h) / 3.0
        
    def f(h):
        return volume_esfera(h) - volume_alvo
        
    # 2. Fase I Explícita: Tabelamento para encontrar intervalos das 3 raízes
    print("--- Fase I: Isolamento (Tabelamento) ---")
    pontos_h = np.linspace(-5, 12, 171) # Malha de passo 0.1
    intervalos = []
    
    for i in range(len(pontos_h) - 1):
        h_atual = pontos_h[i]
        h_prox = pontos_h[i+1]
        if f(h_atual) * f(h_prox) <= 0:
            intervalos.append((h_atual, h_prox))
            
    for i, (a, b) in enumerate(intervalos):
        print(f"Raiz {i+1} isolada no intervalo: [{a:.2f}, {b:.2f}]")

    # A.1 e A.2: Encontrando as raízes (3. Justificativa do Método)
    print("\n--- A.1 e A.2: Cálculo das Três Raízes Reais ---")
    print("Justificativa: Como a Fase I garantiu intervalos isolados com mudança de sinal,")
    print("a Bissecção foi escolhida por sua garantia absoluta de convergência.")
    
    raizes = []
    h_fisico = None
    iteracoes_fisica = 0
    
    for i, (a, b) in enumerate(intervalos):
        # Executa a bissecção e colhe os dados para o relatório
        raiz, hist = bisseccao(f, a, b, eps=eps_padrao, max_iter=max_iter_padrao)
        raizes.append(raiz)
        iteracoes = len(hist)
        chamadas_f = hist[-1]['chamadas_f']
        
        print(f"Raiz {i+1}: {raiz:.6f} m (Iterações: {iteracoes} | Avaliações f: {chamadas_f})")
        
        # O tanque vai de h=0 até h=2R (6m)
        if 0 <= raiz <= (2 * R):
            h_fisico = raiz
            iteracoes_fisica = iteracoes
            
    # 4. Reportar resultado com unidade física
    print(f"\nResultado Físico: h = {h_fisico:.4f} m (encontrado em {iteracoes_fisica} iterações)")
    
    # 5. Verificação substituindo de volta no problema original
    v_verificacao = volume_esfera(h_fisico)
    print(f"Verificação: V({h_fisico:.4f}) = {v_verificacao:.4f} m³ (Alvo era {volume_alvo} m³)")

    # A.3: Tabela h x V e Gráfico
    print("\n--- A.3: Tabela h x V (Volumes de 10 a 110 m³) ---")
    dados_tabela = []
    
    for v_alvo in range(10, 120, 10):
        # Cria uma função dinâmica para cada volume alvo
        f_v = lambda h, v_alvo=v_alvo: volume_esfera(h) - v_alvo
        # Intervalo físico fixo [0, 6] cobre qualquer volume válido do tanque
        raiz_h, hist_h = bisseccao(f_v, 0.0, 6.0, eps=eps_padrao, max_iter=max_iter_padrao)
        dados_tabela.append({
            "Volume_V_m3": v_alvo, 
            "Altura_h_m": round(raiz_h, 4),
            "Iteracoes": len(hist_h)
        })
        
    df_tabela = pd.DataFrame(dados_tabela)
    print(df_tabela.to_string(index=False))

    # Geração do gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(df_tabela["Volume_V_m3"], df_tabela["Altura_h_m"], marker="o", color='b')
    plt.title("Problema A - Relação Altura x Volume no Reservatório Esférico")
    plt.xlabel("Volume V (m³)")
    plt.ylabel("Altura h (m)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figuras/problema_A_h_vs_V.png", dpi=180)
    plt.close()
    
    print("\nGráfico salvo com sucesso em 'figuras/problema_A_h_vs_V.png'")

# Problema B
def problema_B():
    print("=== Problema B - Perda de carga (Colebrook-White) ===")

    # Dados do problema
    D = 0.100       # Diâmetro interno (m)
    rug = 4.5e-5    # Rugosidade absoluta (m)
    Re = 2.0e5      # Número de Reynolds
    L = 500.0       # Comprimento (m)
    Q = 0.050       # Vazão (m3/s)
    g = 9.81        # Gravidade (m/s2)
    
    eps_padrao = 1e-8 # Para garantir 6 casas decimais precisas
    
    # Constantes auxiliares para simplificar a equação
    A_const = rug / (3.7 * D)
    B_const = 2.51 / Re

    # 1. Dedução da função f(x), onde x é o fator de atrito (f)
    def F(x):
        if x <= 0:
            raise ValueError("Fator de atrito deve ser positivo para a raiz quadrada.")
        return (1.0 / math.sqrt(x)) + 2.0 * math.log10(A_const + B_const / math.sqrt(x))

    # Derivada analítica de f(x) (Complexa e propensa a erros)
    def dF(x):
        termo_log = A_const + B_const / math.sqrt(x)
        parte1 = -0.5 * (x ** -1.5)
        parte2 = 2.0 * (1.0 / math.log(10)) * (1.0 / termo_log) * B_const * (-0.5 * (x ** -1.5))
        return parte1 + parte2

    # 2. Fase I: Isolamento
    print("\n--- Fase I: Isolamento ---")
    xs = np.linspace(0.01, 0.08, 71) # Valores típicos de fator de atrito
    intervalos = []
    for i in range(len(xs) - 1):
        if F(xs[i]) * F(xs[i+1]) <= 0:
            intervalos.append((xs[i], xs[i+1]))
    print(f"Raiz isolada no intervalo: {intervalos[0]}")
    a_iso, b_iso = intervalos[0]

    # B.1 e B.2: Comparação dos Métodos e Dificuldade Analítica
    print("\n--- B.1 e B.2: Cálculo de f com 6 casas decimais e Comparação ---")
    
    # Bissecção
    raiz_b, hist_b = bisseccao(F, a_iso, b_iso, eps=eps_padrao)
    print(f"Bissecção: f = {raiz_b:.6f} | Iterações: {len(hist_b)} | Avaliações f: {hist_b[-1]['chamadas_f']}")
    
    # Secante (Chutes arbitrários)
    try:
        raiz_s, hist_s = secante(F, 0.05, 0.02, eps=eps_padrao)
        print(f"Secante (x0=0.05, x1=0.02): f = {raiz_s:.6f} | Iterações: {len(hist_s)} | Avaliações f: {hist_s[-1]['chamadas_f']}")
    except Exception as e:
        print(f"Secante falhou: {e}")

    # Newton (Tentativa com chute arbitrário x0=0.05 sugerido no enunciado)
    try:
        raiz_n, hist_n = newton(F, dF, 0.05, eps=eps_padrao)
        print(f"Newton (x0=0.05): f = {raiz_n:.6f} | Iterações: {len(hist_n)}")
    except Exception as e:
        print(f"Newton (x0=0.05): FALHOU - Erro interceptado: {e}")
        print("-> Justificativa: A tangente projetou o próximo x para um valor negativo, saindo do domínio da raiz quadrada.")

    # B.3: Chute de Swamee-Jain
    print("\n--- B.3: Chute inicial de Swamee-Jain ---")
    f_sj = 0.25 / (math.log10(rug/(3.7*D) + 5.74/(Re**0.9))**2)
    print(f"Valor do chute Swamee-Jain (f0): {f_sj:.6f}")

    raiz_n_sj, hist_n_sj = newton(F, dF, f_sj, eps=eps_padrao)
    raiz_s_sj, hist_s_sj = secante(F, f_sj, f_sj*1.05, eps=eps_padrao)
    
    print(f"Newton c/ SJ: f = {raiz_n_sj:.6f} | Iterações: {len(hist_n_sj)}")
    print(f"Secante c/ SJ: f = {raiz_s_sj:.6f} | Iterações: {len(hist_s_sj)}")

    # B.4: Equação de Darcy-Weisbach
    print("\n--- B.4 e B.5: Perda de carga (hf) e Análise de Sensibilidade ---")
    f_exato = raiz_b
    area = math.pi * (D**2) / 4.0
    V = Q / area
    
    hf_exato = f_exato * (L / D) * (V**2 / (2 * g))
    print(f"5. Verificação / B.4: Velocidade = {V:.4f} m/s | Perda de carga (hf real) = {hf_exato:.4f} m")

    # B.5: Fator arredondado para 0.02
    f_aprox = 0.02
    hf_aprox = f_aprox * (L / D) * (V**2 / (2 * g))
    erro_pct = abs(hf_aprox - hf_exato) / hf_exato * 100
    
    print(f"Sensibilidade (f=0.02): Perda de carga (hf aprox) = {hf_aprox:.4f} m")
    print(f"Erro percentual em hf: {erro_pct:.2f}%")

# Problema C
def problema_C():
    print("=== Problema C - Gás Ideal vs van der Waals ===")

    # Dados do problema (unidades do SI)
    R = 8.314       # Constante dos gases (J/(mol K))
    T = 300.0       # Temperatura (K)
    a = 0.3640      # Constante a (Pa m^6 / mol^2)
    b = 4.267e-5    # Constante b (m^3 / mol)
    P = 5.0e6       # Pressão (Pa) - 5 MPa

    eps_padrao = 1e-13 # Alta precisão para termodinâmica
    
    # 1. C.1: Volume pelo modelo de Gás Ideal
    v_ideal = R * T / P
    print(f"C.1: Volume pelo modelo de Gás Ideal (v_ideal) = {v_ideal:.6e} m3/mol")

    # Dedução da função f(v) a partir de (P + a/v^2)(v - b) = RT
    # Multiplicando todos os termos por v^2 e rearranjando, obtemos a cúbica:
    # P*v^3 - (P*b + R*T)*v^2 + a*v - a*b = 0
    def F(v, pressao=P):
        return pressao * (v**3) - (pressao * b + R * T) * (v**2) + a * v - a * b

    def dF(v, pressao=P):
        return 3 * pressao * (v**2) - 2 * (pressao * b + R * T) * v + a

    # 2. Fase I: Isolamento (Tabelamento)
    print("\n--- Fase I: Isolamento (Tabelamento) ---")
    # O volume deve ser estritamente maior que 'b' (volume ocupado pelas próprias moléculas). 
    # Vamos testar na vizinhança entre 'b' e o dobro do volume ideal.
    xs = np.linspace(b * 1.1, v_ideal * 2, 100)
    intervalos = []
    for i in range(len(xs) - 1):
        if F(xs[i]) * F(xs[i+1]) <= 0:
            intervalos.append((xs[i], xs[i+1]))
            
    print(f"Raiz isolada no intervalo: [{intervalos[0][0]:.6e}, {intervalos[0][1]:.6e}]")

    # C.2 e C.3: Escolha do método, chute inicial e cálculo
    print("\n--- C.2 e C.3: Cálculo do volume real (v_vdw) ---")
    print("Justificativa: Como temos uma estimativa física excelente (o volume do gás ideal),")
    print("escolhemos o método de Newton usando v_ideal como chute inicial (x0).")
    
    raiz_n, hist_n = newton(F, dF, v_ideal, eps=eps_padrao)
    print(f"Newton (x0=v_ideal): v_vdw = {raiz_n:.6e} m3/mol | Iterações: {len(hist_n)}")
    
    # 5. Verificação substituindo na equação original
    # (P + a/v^2)(v - b) = RT -> lado esquerdo deve ser igual a RT (8.314 * 300 = 2494.2)
    lado_esquerdo = (P + a / (raiz_n**2)) * (raiz_n - b)
    lado_direito = R * T
    print(f"Verificação (Equação de Estado): Lado Esquerdo = {lado_esquerdo:.4f} | Lado Direito = {lado_direito:.4f}")

    # Erro em relação ao gás ideal
    erro_pct = abs(v_ideal - raiz_n) / raiz_n * 100
    print(f"Erro do modelo ideal em relação ao real: {erro_pct:.2f}%")

    # C.4: Isoterma (Gráfico P x v)
    print("\n--- C.4: Geração da Isoterma (1 a 10 MPa) ---")
    dados_tabela = []
    
    for P_MPa in range(1, 11): # 1 a 10 MPa
        pressao_atual = P_MPa * 1e6
        v_id_atual = R * T / pressao_atual
        
        # Usa Newton com v_ideal atual como chute para achar o v real naquela pressão específica
        v_real_atual, _ = newton(lambda v: F(v, pressao_atual), lambda v: dF(v, pressao_atual), v_id_atual, eps=1e-13)
            
        dados_tabela.append({
            "Pressao (MPa)": P_MPa,
            "v_ideal (m3/mol)": f"{v_id_atual:.4e}",
            "v_real_vdw (m3/mol)": f"{v_real_atual:.4e}"
        })
        
    df_isoterma = pd.DataFrame(dados_tabela)
    print(df_isoterma.to_string(index=False))

    # Geração do gráfico
    v_ideais = [float(x["v_ideal (m3/mol)"]) for x in dados_tabela]
    v_reais = [float(x["v_real_vdw (m3/mol)"]) for x in dados_tabela]
    pressoes = [x["Pressao (MPa)"] for x in dados_tabela]

    plt.figure(figsize=(8, 5))
    plt.plot(v_reais, pressoes, marker="o", color="blue", label="van der Waals (Real)")
    plt.plot(v_ideais, pressoes, marker="s", color="red", linestyle="--", label="Gás Ideal")
    plt.title("Isoterma T = 300 K (van der Waals vs Gás Ideal)")
    plt.xlabel("Volume Molar v (m3/mol)")
    plt.ylabel("Pressão P (MPa)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figuras/problema_C_isoterma.png", dpi=180)
    plt.close()
    
    print("\nGráfico salvo com sucesso em 'figuras/problema_C_isoterma.png'")


# Problema D
def problema_D():
    print("=== Problema D - Taxa Interna de Retorno (TIR) ===")

    # Primeiro fluxo de caixa
    fluxos_1 = [-1000.0, 300.0, 350.0, 400.0, 450.0]
    eps_padrao = 1e-8

    # 1. Dedução da função VPL(i)
    def vpl_1(i):
        if i <= -1.0:
            raise ValueError("A taxa i deve ser maior que -100%.")
        return sum(c / ((1.0 + i) ** k) for k, c in enumerate(fluxos_1))

    # 2. Fase I: Isolamento (Tabelamento para a taxa i entre 0% e 50%)
    print("\n--- Fase I: Isolamento (Primeiro fluxo de caixa) ---")
    xs = np.linspace(0.0, 0.5, 51)
    intervalos = []
    for j in range(len(xs) - 1):
        if vpl_1(xs[j]) * vpl_1(xs[j+1]) <= 0:
            intervalos.append((xs[j], xs[j+1]))
            
    print(f"Intervalo com mudança de sinal isolado: {intervalos[0]}")

    # 3. Cálculo da TIR usando Bissecção
    print("\n--- D.1 e D.2: Cálculo da TIR (Primeiro fluxo de caixa) ---")
    a_i, b_i = intervalos[0]
    tir_val, hist_tir = bisseccao(vpl_1, a_i, b_i, eps=eps_padrao)
    print(f"TIR encontrada = {tir_val:.6f} ({tir_val*100:.2f}%) | Iterações: {len(hist_tir)} | Avaliações f: {hist_tir[-1]['chamadas_f']}")

    # 4. Avaliação do VPL a 15% e 20%
    vpl_15 = vpl_1(0.15)
    vpl_20 = vpl_1(0.20)
    print(f"VPL a 15% (0.15) = R$ {vpl_15:.4f}")
    print(f"VPL a 20% (0.20) = R$ {vpl_20:.4f}")

    # 5. Verificação substituindo de volta
    vpl_na_tir = vpl_1(tir_val)
    print(f"Verificação: VPL(TIR) = {vpl_na_tir:.2e} (Deveria ser 0)")

    # Segundo fluxo de caixa: Múltiplas mudanças de sinal
    print("\n--- D.3: Segundo fluxo de caixa (Múltiplas Mudanças de Sinal) ---")
    fluxos_2 = [-1000.0, 2500.0, -1540.0]
    def vpl_2(i):
        if i <= -1.0:
            raise ValueError("Taxa inválida.")
        return sum(c / ((1.0 + i) ** k) for k, c in enumerate(fluxos_2))

    # Usando a malha corrigida para não pisar exatamente na raiz e gerar ValueError
    xs_2 = np.linspace(0.015, 0.585, 60)
    intervalos_2 = []
    for j in range(len(xs_2) - 1):
        if vpl_2(xs_2[j]) * vpl_2(xs_2[j+1]) <= 0:
            intervalos_2.append((xs_2[j], xs_2[j+1]))
            
    print(f"Intervalos com mudança de sinal no Segundo fluxo: {intervalos_2}")
    for idx, (lo, hi) in enumerate(intervalos_2):
        r_sub, hist_sub = bisseccao(vpl_2, lo, hi, eps=eps_padrao)
        print(f"TIR {idx+1} do Segundo fluxo = {r_sub:.6f} ({r_sub*100:.2f}%) | Iterações: {len(hist_sub)}")

    # Gráfico do VPL (Segundo fluxo de caixa)
    print("\n--- Geração do Gráfico do VPL (Segundo fluxo de caixa) ---")
    taxas_plot = np.linspace(0.0, 0.6, 100)
    vpl_plot = [vpl_2(tx) for tx in taxas_plot]

    plt.figure(figsize=(8, 5))
    plt.plot(taxas_plot, vpl_plot, color='green', linewidth=2)
    plt.axhline(0, color='red', linestyle='--') # Linha horizontal no VPL = 0
    plt.title("Problema D - VPL vs Taxa de Desconto (Segundo Fluxo)")
    plt.xlabel("Taxa de Desconto i (Decimal)")
    plt.ylabel("VPL (R$)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figuras/problema_D_vpl.png", dpi=180)
    plt.close()
    
    print("Gráfico salvo com sucesso em 'figuras/problema_D_vpl.png'")

# Problema E
def problema_E():
    print("=== Problema E - Equação de Kepler ===")

    eps_padrao = 1e-8

    # Função F(E) e Derivada dF(E) genéricas
    def F(E, e, M):
        return E - e * math.sin(E) - M

    def dF(E, e):
        return 1.0 - e * math.cos(E)

    # E.1: Cometa Halley
    print("\n--- E.1: Cálculo para o Cometa Halley ---")
    e_halley = 0.967
    M_halley = 0.2

    # Fase I: Isolamento (Sabemos que E deve estar entre 0 e pi)
    print("Fase I: Isolamento no intervalo [0, pi]")
    xs = np.linspace(0.0, math.pi, 32)
    intervalos = []
    for i in range(len(xs) - 1):
        if F(xs[i], e_halley, M_halley) * F(xs[i+1], e_halley, M_halley) <= 0:
            intervalos.append((xs[i], xs[i+1]))
            
    print(f"Raiz isolada no intervalo: {intervalos[0]}")

    # Cálculo por Newton (chute inicial E0 = M, conforme sugerido nos próximos itens)
    raiz_h, hist_h = newton(lambda E: F(E, e_halley, M_halley), 
                            lambda E: dF(E, e_halley), 
                            x0=M_halley, eps=eps_padrao)
                            
    print(f"Resultado E.1: E = {raiz_h:.6f} rad | Iterações: {len(hist_h)} | Avaliações f: {hist_h[-1]['chamadas_f']}")
    print(f"Verificação: E - e*sin(E) = {(raiz_h - e_halley * math.sin(raiz_h)):.4f} (Deve ser igual a M = {M_halley})")

    # E.2: Robustez (Método de Newton)
    print("\n--- E.2: Teste de Robustez com e -> 1 ---")
    casos = [
        (0.10, 0.50, "i"),
        (0.90, 0.10, "ii"),
        (0.99, 0.01, "iii")
    ]

    print("Caso | e      | M (rad) | E (rad)  | Iteracoes | f'(E0)")
    for e_c, M_c, nome in casos:
        r, h = newton(lambda E: F(E, e_c, M_c), lambda E: dF(E, e_c), x0=M_c, eps=eps_padrao)
        df_inicio = dF(M_c, e_c)
        print(f"{nome:>4} | {e_c:.2f}   | {M_c:.2f}    | {r:.6f} | {len(h):<9} | {df_inicio:.6f}")

    # E.3: Chute Inicial Melhorado
    print("\n--- E.3: Chute inicial melhorado no Caso (iii) ---")
    e_3, M_3 = 0.99, 0.01
    chute_melhor = M_3 + e_3 * math.sin(M_3)
    
    r_3, h_3 = newton(lambda E: F(E, e_3, M_3), lambda E: dF(E, e_3), x0=chute_melhor, eps=eps_padrao)
    print(f"Valor do novo chute (E0): {chute_melhor:.6f} rad")
    print(f"Newton c/ chute melhorado: E = {r_3:.6f} rad | Iterações: {len(h_3)}")

    # E.4: Comparação com Bissecção
    print("\n--- E.4: Bissecção no Caso (iii) ---")
    r_b, h_b = bisseccao(lambda E: F(E, e_3, M_3), 0.0, math.pi, eps=eps_padrao)
    print(f"Bissecção [0, pi]: E = {r_b:.6f} rad | Iterações: {len(h_b)}")

# Problema F
def problema_F():
    print("=== Problema F - Deflexão de viga ===")

    # Constantes físicas do problema
    L = 600.0
    E = 50000.0
    I = 30000.0
    w0 = 2.5
    eps_padrao = 1e-8

    # Isolando as constantes (C) para simplificar a função na memória
    C = w0 / (120.0 * L * E * I)

    # Deflexão y(x)
    def y(x):
        return C * (- (x**5) + 2.0 * (L**2) * (x**3) - (L**4) * x)

    # Derivada dy/dx (Analítica)
    def dy_dx(x):
        return C * (-5.0 * (x**4) + 6.0 * (L**2) * (x**2) - (L**4))

    # F.1 e F.2 Ponto de deflexão máxima (Analítico)
    # Fazendo a substituição u = x^2 na derivada, a raiz válida cai em u = 0.2 * L^2
    x_max_analitico = L * math.sqrt(0.2)
    print("\n--- F.1 e F.2: Derivada Analítica e Deflexão Máxima ---")
    print(f"Ponto de deflexão máxima (x): {x_max_analitico:.6f} cm")
    
    deflexao_max = y(x_max_analitico)
    print(f"Deflexão máxima y(x): {deflexao_max:.6f} cm")

    # F.3 Armadilha Proposital
    print("\n--- F.3: Armadilha Proposital (Avaliação da Bissecção) ---")
    dy_0 = dy_dx(0.0)
    dy_L = dy_dx(L)
    print(f"dy/dx em x=0: {dy_0:.6e}")
    print(f"dy/dx em x=L: {dy_L:.6e} (Virtualmente ZERO)")
    
    try:
        bisseccao(dy_dx, 0.0, L, eps=eps_padrao)
    except Exception as e:
        print(f"Teste Bissecção [0, L]: FALHOU CORRETAMENTE (Proteção Ativada!) - Erro: {e}")
        
    print("\nCorreção: Rodando no intervalo isolado [0, 300] (metade da viga):")
    raiz_f, hist_f = bisseccao(dy_dx, 0.0, 300.0, eps=eps_padrao)
    print(f"Raiz encontrada: x = {raiz_f:.6f} cm | Iterações: {len(hist_f)}")

    # F.4 Derivada Numérica e o limite de passo h
    print("\n--- F.4: Raízes Numéricas e Fenômeno do Passo h ---")
    hs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    erros = []
    
    print(" h        | Raiz Numérica (cm) | Erro Absoluto")
    for h in hs:
        # Criamos uma derivada numérica oem fluxo baseada no tamanho do h atual
        def dy_num(x):
            return (y(x + h) - y(x - h)) / (2.0 * h)
            
        # Calcula a nova raiz usando a função derivada numérica dentro da bissecção
        raiz_num, _ = bisseccao(dy_num, 0.0, 300.0, eps=eps_padrao)
        erro = abs(raiz_num - x_max_analitico)
        erros.append(erro)
        print(f"{h:.1e} | {raiz_num:.6f}         | {erro:.6e}")
        
    # Gráfico do Comportamento do Erro
    plt.figure(figsize=(8, 5))
    plt.plot(hs, erros, marker="s", color="red", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis() # Inverte para mostrar o h diminuindo para a direita
    plt.title("Problema F - Efeito da Precisão de Máquina na Derivada Numérica")
    plt.xlabel("Tamanho do passo numérico h (log)")
    plt.ylabel("Erro Absoluto na localização de x (log)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figuras/problema_F_erro_h.png", dpi=180)
    plt.close()
    
    print("\nGráfico salvo em 'figuras/problema_F_erro_h.png'")

# Execução Principal
if __name__ == "__main__":
    problema_A()
    problema_B()
    problema_C()
    problema_D()
    problema_E()
    problema_F()