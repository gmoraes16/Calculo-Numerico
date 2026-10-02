import numpy as np

# Função fornecida pelo professor 
def newton(F, JF, x0, TOL, N):  
    #preliminares  
    x = np.copy(x0).astype('double')  
    k = 0  
    #iteracoes  
    while (k < N):  
       k += 1  
       #iteracao Newton  
       delta = -np.linalg.inv(JF(x)).dot(F(x))  
       x = x + delta  
       #criterio de parada  
       if (np.linalg.norm(delta,np.inf) < TOL):  
           return x  
 
    raise NameError('num. max. iter. excedido.')

# Vetor de Funções F(X) = 0
def F(vars):
    x, y, z = vars[0], vars[1], vars[2]
    return np.array([
        6*x - 2*y + np.exp(z) - 2,
        np.sin(x) - y + z,
        np.sin(x) + 2*y + 3*z - 1
    ])

# Matriz Jacobiana
def JF(vars):
    x, y, z = vars[0], vars[1], vars[2]
    return np.array([
        [6, -2, np.exp(z)],
        [np.cos(x), -1, 1],
        [np.cos(x), 2, 3]
    ])

# Parâmetros e chute inicial
TOL = 1e-5
N = 50
x0 = np.array([0.0, 0.0, 0.0])

# Executa o método
raiz = newton(F, JF, x0, TOL, N)

# Imprime os resultados
print("Aproximação da raiz do sistema:")
print(f"x = {raiz[0]:.6f}\ny = {raiz[1]:.6f}\nz = {raiz[2]:.6f}")