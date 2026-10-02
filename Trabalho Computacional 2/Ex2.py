import numpy as np

# Variáveis do problema
R1 = R2 = R3 = 100
R4 = R5 = 150
R6 = R7 = R8 = 200
V1 = 10
V2 = 12

# Montando a matriz enunciada
R = np.array([
    [R1 + R2 + R4, -R2, 0, -R4],
    [-R2, R2 + R3 + R5, -R5, 0],
    [0, -R5, R5 + R7 + R8, -R7],
    [-R4, 0, -R7, R4 + R6 + R7]
])

# Vetor de tensões
V = np.array([-V1, V2, 0, 0])

# Resolve o sistema (R * i = V)
i = np.linalg.solve(R, V)

print("Correntes do circuito (em Amperes):")
for j in range(4):
    print(f"i{j+1} = {i[j]:.4f} A")