# -*- coding: utf-8 -*-
"""
Script Python para explorar o modelo de regressão linear do Capítulo 2.
"""

# Biblioteca de matemática
import numpy as np
# Biblioteca de plotagem
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# DADOS
# ----------------------------------------------------------------------------
# Cria alguns dados de entrada / saída
x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
y = np.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6])

print("Dados de entrada (x):")
print(x)
print("\nDados de saída (y):")
print(y)

# ----------------------------------------------------------------------------
# DEFINIÇÃO DAS FUNÇÕES
# ----------------------------------------------------------------------------

def f(x, phi0, phi1):
  """
  Define o modelo de regressão linear 1D.
  """
  # TODO: Substitua esta linha pelo modelo de regressão linear (equação 2.4 do livro)
  # A equação é: y = phi0 + phi1 * x
  y_pred = phi0 + phi1 * x

  return y_pred

def compute_loss(x, y, phi0, phi1):
  """
  Calcula a perda (soma dos erros quadrados).
  """

  # TODO: Substitua esta linha pelo cálculo da perda (equação 2.5 do livro)
  # A equação é: L(phi0, phi1) = sum((y_i - f(x_i, phi0, phi1))^2) para todos os i

  total_loss = 0
  points_number = len(x)

  for i in range(points_number):
    current_x = x[i]
    current_y = y[i]

    predicted_y = phi0 + phi1 * current_x
    squared_error = (current_y - predicted_y)**2

    total_loss += squared_error

  return total_loss

def plot(x, y, phi0, phi1, title=""):
    """
    Função auxiliar para plotar os dados e a linha do modelo.
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlim([0, 2.0])
    plt.ylim([0, 2.0])
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y$')
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Desenha a linha
    x_line = np.arange(0, 2, 0.01)
    y_line = f(x_line, phi0, phi1)
    plt.plot(x_line, y_line, 'b-', lw=2)
    
    plt.show()

# ----------------------------------------------------------------------------
# EXECUÇÃO DO EXERCÍCIO
# ----------------------------------------------------------------------------

# --- Parte 1: Modelo inicial (Figura 2.2b) ---
print("\n--- Parte 1: Testando o primeiro modelo ---")
phi0 = 0.4
phi1 = 0.2
plot(x, y, phi0, phi1, "Modelo Inicial (fig 2.2b)")
loss = compute_loss(x, y, phi0, phi1)
print(f'Sua Perda = {loss:.2f}')

# --- Parte 2: Segundo modelo (Figura 2.2c) ---
print("\n--- Parte 2: Testando o segundo modelo ---")
phi0 = 1.60
phi1 = -0.8
plot(x, y, phi0, phi1, "Segundo Modelo (fig 2.2c)")
loss = compute_loss(x, y, phi0, phi1)
print(f'Sua Perda = {loss:.2f}')

# --- Parte 3: Ajuste manual dos parâmetros ---
print("\n--- Parte 3: Ajuste seus parâmetros aqui! ---")
# TODO: Mude os parâmetros manualmente para ajustar o modelo.
# 1. Fixe phi1 e tente mudar phi0 até que a perda não diminua mais.
# 2. Depois, fixe phi0 e tente mudar phi1 até que a perda não diminua mais.
# Repita o processo até encontrar um bom ajuste (como na figura 2.2d).
# Comece com estes valores:
phi0 = 0.84  # Mude este valor
phi1 = 0.48  # Mude este valor

plot(x, y, phi0, phi1, "Seu Melhor Ajuste Manual")
print(f'Sua Perda Final = {compute_loss(x, y, phi0, phi1):.2f}')


# ----------------------------------------------------------------------------
# VISUALIZAÇÃO DA FUNÇÃO DE PERDA
# ----------------------------------------------------------------------------

print("\n--- Visualizando a superfície de perda ---")
# Cria uma grade 2D de possíveis valores de phi0 e phi1
phi0_mesh, phi1_mesh = np.meshgrid(np.arange(0.0, 2.0, 0.02), np.arange(-1.0, 1.0, 0.02))

# Cria um array 2D para as perdas
all_losses = np.zeros_like(phi1_mesh)

# Itera sobre cada combinação de phi0, phi1 e calcula a perda
# (Isso pode demorar um pouco se a sua função de perda não estiver implementada)
for indices, temp in np.ndenumerate(phi1_mesh):
  all_losses[indices] = compute_loss(x, y, phi0_mesh[indices], phi1_mesh[indices])

# Plota a função de perda como um mapa de calor
fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(7, 7)
levels = 256
ax.contourf(phi0_mesh, phi1_mesh, all_losses, levels)
levels = 40
ax.contour(phi0_mesh, phi1_mesh, all_losses, levels, colors=['#80808080'])
ax.set_ylim([1, -1])
ax.set_xlabel(r'Intercept, $\phi_0$')
ax.set_ylabel(r'Slope, $\phi_1$')
ax.set_title('Superfície de Perda')

# Plota a posição da sua melhor linha de ajuste na função de perda
# Deve estar perto do mínimo (a área mais escura)
ax.plot(phi0, phi1, 'ro', label=f'Seu Ponto ($\phi_0$={phi0:.2f}, $\phi_1$={phi1:.2f})')
ax.legend()
plt.show()