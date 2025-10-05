import cv2
from matplotlib import pyplot as plt
import numpy as np

def filter2D_pixelwise(image, kernel, coords):
    blurred_pixel = [0,0,0]
    window_dim = kernel.shape
    y,x = coords
    for u in np.arange(0, window_dim[0]):
        for v in np.arange(0, window_dim[1]):
            current_coords = (np.uint((y+u) - (window_dim[0]-1)/2), np.uint((x+v) - (window_dim[0]-1)/2))
            blurred_pixel += (image[current_coords])*kernel[u, v]
    return blurred_pixel

# Parâmetros do algoritmo de suavização
tattoo_region = [[70, 227],[607, 462]] # Um array que define a região retangular que será suavizada
                      # É definida pelos pixels superior esquerdo e inferior direito

border_tau = [150, 40]

window_dimension = (3,29) # Tamanho da janela quadrada da filtragem 2d

kernel = np.ones(window_dimension, np.float32)/(window_dimension[0]*window_dimension[1]) # Define o kernel da filtragem 2d

# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/Tatoo1.jpg')

# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# ----------------------------------------------------------
# Algoritmo de suavização

# Isolamento da região de interesse que se deseja suavizar
I_tattoo = I_input[tattoo_region[0][0]:tattoo_region[1][0], tattoo_region[0][1]:tattoo_region[1][1]]

# Detecção de bordas
I_border = cv2.Canny(I_tattoo, border_tau[0], border_tau[1])

# Aplicação da suavização nas regiões de borda
I_tattoo_lines, I_tattoo_columns, _ = I_tattoo.shape
for y in np.arange(1, I_tattoo_lines-1):
    for x in np.arange(1, I_tattoo_columns-1):
        blur_pixel = False
        for u in np.arange(-1,2):
            for v in np.arange(-1,2):
                blur_pixel = blur_pixel or I_border[y+u,x+v]
        if blur_pixel:
            I_tattoo[y,x] = filter2D_pixelwise(I_tattoo, kernel, (y,x))

I_output = I_tattoo

# Exibição das imagens
ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem original')

ax2.imshow(I_border, cmap='gray')
ax2.set_title('Imagem de borda')

# ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
# ax2.set_title('Imagem suavizada')

plt.show()