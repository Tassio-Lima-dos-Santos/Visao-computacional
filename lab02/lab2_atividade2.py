import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

# Algoritmo de Canny
def CannyBorderDetection(I_input, sigma):

    n_linhas, n_colunas = I_input.shape

    # Primeira Etapa
    w = sigma*6 + 1
    K_gauss = visco.gaussianKernel(w, sigma)
    I_blur = cv2.filter2D(I_input, -1, K_gauss, borderType=cv2.BORDER_REFLECT101)

    # Segunda Etapa

    # Kernel da derivada parcial de x
    Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
    # Kernel da derivada parcial de y
    Ky = np.transpose(Kx)
    
    # Imagem da derivada parcial em respeito a x da imagem suavizada
    Ix = cv2.filter2D(I_blur, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
    # Imagem da derivada parcial em respeito a y da imagem suavizada
    Iy = cv2.filter2D(I_blur, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)

    # Imagem da magnitude do gradiente
    I_gradient_length = np.sqrt(Ix**2 + Iy**2)
    I_gradient_angle = np.arctan2(Iy,Ix)*(180/np.pi)

    # Terceira Etapa

    # Etapa 3.a
    Gs = np.zeros(I_blur.shape, np.float32)

    for v in np.arange(1, n_colunas):
        for u in np.arange(1, n_linhas): # Os loops for começam em 1 para ignorar a borda da imagem
            # Etapa 3.b
            theta = I_gradient_angle[u,v]

            # Etapa 3.c
            ''' É criada uma variável direção que mapeia o ângulo a um valor entre -4 e 4
            A variável direção sempre é inteira, pois surge de um arredondamento
            Os ângulos entre -180 e -157,5 são mapeados para -4, os ângulos entre -157,5 
            e -112,5 são mapeados para -3 e assim por diante. Essa variável foi criada 
            para permitir o uso de switch case'''
            direction = np.around(theta/45)
            match direction:
                case 1 | -3:
                    comp_pixels = [I_gradient_length[u-1,v-1], I_gradient_length[u+1,v+1]]
                    pass
                case 2 | -2:
                    comp_pixels = [I_gradient_length[u-1,v], I_gradient_length[u+1,v]]
                    pass
                case 3 | -1:
                    comp_pixels = [I_gradient_length[u-1,v+1], I_gradient_length[u+1,v-1]]
                    pass
                case _:
                    comp_pixels = [I_gradient_length[u,v-1], I_gradient_length[u,v+1]]
                    pass

    # Quarta Etapa

    # Quinta Etapa


    return I_output

# ----------------------------------------------------------
# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/castle.jpg', cv2.IMREAD_GRAYSCALE)

I_output = CannyBorderDetection(I_input, 2)

# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)

# Exibição das imagens
ax1.imshow(I_input, cmap='gray')
ax1.set_title('Imagem original')

im2 = ax2.imshow(I_output, cmap='gray')
ax2.set_title('Imagem de bordas')

plt.show()