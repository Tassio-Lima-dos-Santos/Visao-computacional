import cv2
from matplotlib import pyplot as plt
import numpy as np

# Parâmetros do algoritmo de suavização
blur_region = np.array([], np.uint8) # Um array que define a região retangular que será suavizada
                                     # É definida pelos pixels superior esquerdo e inferior direito
window_size = 5 # Tamanho da janela quadrada da filtragem 2d
kernel = # Define o kernel da filtragem 2d

# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/Tatoo1.jpg')

# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)

# ----------------------------------------------------------
# Algoritmo de suavização

# Exibição das imagens
ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem original')

ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagem suavizada')