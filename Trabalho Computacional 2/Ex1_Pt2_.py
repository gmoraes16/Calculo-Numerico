import numpy as np

g = 9.81
z = np.array([100.0, 85.0, 60.0])
L = np.array([1200.0, 900.0, 1500.0])
D = np.array([0.30, 0.25, 0.25])
f = np.array([0.022, 0.024, 0.024])
K = 8*f*L/(np.pi**2*g*D**5) # a) Calcule K1, K2, K3
q = 0.200

def F(x):
    Q, H = x[:3], x[3]
    r = np.empty(4)
    r[:3] = K*Q*np.abs(Q) + H - z # Bernoulli em cada adutora
    r[3] = Q.sum() - q # Continuidade no J
    return r

# a) Matriz Jacobiana
def J(x):
    Q = x[:3] # Separa as três vazões

    mat_J = np.zeros((4, 4)) # Cria uma matriz 4x4 preenchida com zeros

    # Preenche a diagonal princiapal (derivadas em relação a Q1, Q2, Q3)
    mat_J[0, 0] = 2*K[0]*np.abs(Q[0])
    mat_J[1, 1] = 2*K[1]*np.abs(Q[1])
    mat_J[2, 2] = 2*K[2]*np.abs(Q[2])

    # Preenche a última coluna (as derivadas de f1, f2, f3 em relação a H)
    mat_J[0:3, 3] = 1.0

    # Preenche a última linha (a derivada de f4 em relação a Q1, Q2, Q3)
    mat_J[3, 0:3] = 1.0

    return mat_J

print("Valores de K:")
print(f"K1 = {K[0]:.4f}")
print(f"K2 = {K[1]:.4f}")
print(f"K3 = {K[2]:.4f}\n")

# b)
def newton_tabela(F, JF, x0, TOL, N, nome_var3="Q3"):
    x = np.copy(x0).astype('double')
    
    # Lista para guardar o valor de x em cada iteração
    historico_x = [np.copy(x)]
    
    k = 0
    convergiu = False

    while (k < N):
        k += 1
        delta = -np.linalg.inv(JF(x)).dot(F(x))
        x = x + delta
        historico_x.append(np.copy(x)) # Guarda o x da iteração atual

        # Critério de Parada
        if (np.linalg.norm(delta, np.inf) < TOL):
            convergiu = True
            break
        
    # O nosso x* é o último elemento gravado no histórico
    x_asterisco = historico_x[-1]
    
    # Imprimindo a tabela pedida em b), mesmo se não convergir
    print(f"{'k':<3} | {'Q1':<9} | {'Q2':<9} | {nome_var3:<9} | {'H':<9} | {'||F(x)||':<12} | {'||x - x*||':<12}")
    print("-" * 75)
    
    # Percorrendo o histórico para imprimir linha a linha
    for i, x_k in enumerate(historico_x):
        norma_F = np.linalg.norm(F(x_k), np.inf)
        erro_x = np.linalg.norm(x_k - x_asterisco, np.inf)
        # Formatação com notação científica (.2e) para os erros, facilitando a análise
        print(f"{i:<3} | {x_k[0]:.6f} | {x_k[1]:.6f} | {x_k[2]:.6f} | {x_k[3]:.5f} | {norma_F:.2e} | {erro_x:.2e}")

    print(f"\n")

    if not convergiu:
        print(f"\n O método parou no limite de {N} iterações.\n")
    return x_asterisco

# Chute inicial fornecido no enunciado
x0 = np.array([0.10, 0.10, -0.05, 70.0]) #
TOL = 1e-6
N = 50

print("Tabela de Iterações - Método de Newton:")
solucao = newton_tabela(F, J, x0, TOL, N)

# c)
# Prova computacional
x0_falha = np.array([0.0, 0.0, 0.0, 70.0])
J_falha = J(x0_falha)
print("Matriz J no ponto de falha:\n", J_falha)
print("Determinante de J:", np.linalg.det(J_falha))
print("Posto de J:", np.linalg.matrix_rank(J_falha))

print("\nTentando Newton com x0 = (0, 0, 0, 70):")
try:
    newton_tabela(F, J, x0_falha, TOL, N)
except np.linalg.LinAlgError as erro:
    print(f"FALHA na iteração 1 -> {type(erro).__name__}: {erro}")
    print("J(x0) é singular (det J = 0): não existe inversa, o passo de Newton não pode ser calculado.")

# d)

# Extraindo Q1, Q2 e Q3 da solução final encontrada
Q_final = solucao[:3]

# Calculando a área da seção transversal
# Fórmula: A = (pi * D^2) / 4
A = (np.pi * D**2) / 4

# Calculando as velocidades
# Fórmula: v = |Q| / A
v = np.abs(Q_final) / A

# Imprimindo os resultados
print("\n--- Resultados ---")
print(f"Vazão Q3 = {Q_final[2]:.6f} m³/s")

print("\nVelocidades calculadas:")
for i in range(3):
    print(f"v{i+1} = {v[i]:.2f} m/s")

# e)
def F_e(vars):
    Q1, Q2, q_dem, H = vars[0], vars[1], vars[2], vars[3]
    return np.array([
        K[0]*Q1*np.abs(Q1) + H - z[0],
        K[1]*Q2*np.abs(Q2) + H - z[1],
        H - z[2], # Equação do reservatório com Q3 = 0
        Q1 + Q2 - q_dem # Equação da continuidade com q variável
    ])

def J_e(vars):
    Q1, Q2, q_dem, H = vars[0], vars[1], vars[2], vars[3]
    mat = np.zeros((4, 4))
    
    # Derivadas em relação a Q1 e Q2
    mat[0, 0] = 2 * K[0] * np.abs(Q1)
    mat[3, 0] = 1.0
    mat[1, 1] = 2 * K[1] * np.abs(Q2)
    mat[3, 1] = 1.0
    
    # Derivada em relação a q
    mat[3, 2] = -1.0
    
    # Derivadas em relação a H
    mat[0, 3] = 1.0
    mat[1, 3] = 1.0
    mat[2, 3] = 1.0
    
    return mat

# Chute inicial para (Q1, Q2, q, H)
# Sabemos que H tem de ser 60 (para z3) e Q1, Q2, q devem ser positivos
x0_e = np.array([0.1, 0.1, 0.2, 60.0])

print("\n--- Tabela de Iterações ---")
solucao_e = newton_tabela(F_e, J_e, x0_e, 1e-6, 50, nome_var3="q*")

# A demanda q* é a terceira variável do vetor
print(f"q* = {solucao_e[2]:.6f} m³/s")

# f)
# Guardar os valores originais para comparação mais tarde
Q1_antigo = solucao[0]
H_antigo = solucao[3]
q_critico_antigo = solucao_e[2]

# Atualizar o fator de atrito da adutora 1 e recalcular o vetor K
f[0] = 0.030
K = 8*f*L/(np.pi**2*g*D**5)

# Recalcular a solução do sistema para encontrar os novos Q1 e H
print("\nRecalculando o sistema original com f1 = 0.030")
solucao_nova = newton_tabela(F, J, x0, 1e-6, 50)
Q1_novo = solucao_nova[0]
H_novo = solucao_nova[3]

# Recalculando para encontrar o novo q*
print("\nRecalculando com f1 = 0.030")
solucao_e_nova = newton_tabela(F_e, J_e, x0_e, 1e-6, 50, nome_var3="q*")
q_critico_novo = solucao_e_nova[2]

# Calculando a variação percentual
var_Q1 = abs(Q1_novo - Q1_antigo) / abs(Q1_antigo) * 100
var_H = abs(H_novo - H_antigo) / abs(H_antigo) * 100
var_q = abs(q_critico_novo - q_critico_antigo) / abs(q_critico_antigo) * 100

print("\n--- Variação ---")
print(f"Novo Q1: {Q1_novo:.5f} m³/s | Variação: {var_Q1:.2f}%")
print(f"Nova Carga H: {H_novo:.5f} m | Variação: {var_H:.2f}%")
print(f"Novo q*: {q_critico_novo:.5f} m³/s | Variação: {var_q:.2f}%")