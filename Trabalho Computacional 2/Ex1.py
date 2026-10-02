import numpy as np

# Importa a tabela como uma matriz
A = np.array([
    [0.70, 0.10, 0.15],
    [0.20, 0.90, 0.10],     
    [0.10, 0.00, 0.75]
])

# Define o vetor com os orçamentos
b = np.array([16, 5, 8 ])

# Define o vetor matrículas
alunos = np.array([4000, 1000, 2000])

# Calcula o custo total destinado aos alunos
custo_total = A @ b 

# Converte para milhão
custo_total_dolares = custo_total * 1_000_000

# Calcula o custo por aluno
custo_por_aluno = custo_total_dolares / alunos

faculdades = ["Sciences", "Engineering", "Computer Science"]

print("Custo anual de educação por aluno:")
for i in range(3):
    # Formata para ter 2 casas decimais
    print(f"{faculdades[i]}: US$ {custo_por_aluno[i]:.2f}")