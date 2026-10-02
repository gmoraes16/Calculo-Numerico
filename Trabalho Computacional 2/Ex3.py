import numpy as np
import matplotlib.pyplot as plt

# Passo 5 e 6
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

# Passo 3
# Definindo f(x) = 0
def F(vars):
    x = vars[0]
    y = vars[1]

    # f1 = x^2 - y + 1
    # f2 = x^2 + y^2/4 - 1
    return np.array([x**2 - y + 1, x**2 + (y**2)/4 - 1])

# Passo 4
# Encontrando a jacobiana
def JF (vars):
    x = vars[0]
    y = vars[1]

    # Matriz com as derivadas parciais que encontramos no passo anterior
    return np.array([
        [2*x, -1],
        [2*x, y / 2]
    ])

# Parametrização
TOL = 1e-5 # Tolerância de erro
N = 50 # Número máximo de iterações

# Passo 1 e 2

# Chute inicial para o ponto no PRIMEIRO quadrante (x positivo)
x0_quad1 = np.array([0.5, 1.5])
raiz_quad1 = newton(F, JF, x0_quad1, TOL, N)

# Chute inicial para o ponto no SEGUNDO quadrante (x negativo)
x0_quad2 = np.array([-0.5, 1.5])
raiz_quad2 = newton(F, JF, x0_quad2, TOL, N)

# Imprimindo os resultados
print("Ponto de Intersecção (1º Quadrante):")
print(f"x = {raiz_quad1[0]:.6f}, y = {raiz_quad1[1]:.6f}\n")

print("Ponto de Intersecção (2º Quadrante):")
print(f"x = {raiz_quad2[0]:.6f}, y = {raiz_quad2[1]:.6f}")

# Esboço das curvas

# Pontos X para as curvas
x_parabola = np.linspace(-2, 2, 400)
x_elipse = np.linspace(-1, 1, 400)

# Equações
y_parabola = x_parabola**2 + 1
y_elipse_positiva = 2 * np.sqrt(1 - x_elipse**2)
y_elipse_negativa = -2 * np.sqrt(1 - x_elipse**2)

# Criando a figura de forma simples
plt.figure(figsize=(6, 6))

plt.plot(x_parabola, y_parabola, label="Parábola", color="blue")
plt.plot(x_elipse, y_elipse_positiva, label="Elipse", color="red")
plt.plot(x_elipse, y_elipse_negativa, color="red")

# Marcando as raízes
plt.scatter([raiz_quad1[0], raiz_quad2[0]], [raiz_quad1[1], raiz_quad2[1]], color="black", zorder=5)

# Estética básica
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 3)

# Salva a imagem
plt.savefig("esboco_curvas.png")
print("Esboço salvo com sucesso como 'esboco_curvas.png'!")