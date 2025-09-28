import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

# Algoritmo de Canny
def CannyBorderDetection(I_input, sigma):
    # Primeira Etapa
    w = sigma*6 + 1
    K_gauss = visco.gaussianKernel(w, sigma)
    I_blur = cv2.filter2D(I_input, -1, K_gauss, borderType=cv2.BORDER_REFLECT101)

    # Segunda Etapa
    
    # Terceira Etapa

    # Quarta Etapa

    # Quinta Etapa

# ----------------------------------------------------------
# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/castle.jpg')

# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)

# Exibição das imagens
ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem original')

ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagem de bordas')