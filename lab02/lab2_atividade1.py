import cv2
from matplotlib import pyplot as plt
import numpy as np

# Parâmetros do algoritmo de apagamento da tatuagem
tattoo_region = [[70, 227],[607, 462]] # Um array que define a região retangular que será processada
                                       # É definida pelos pixels superior esquerdo e inferior direito da região

border_tau = [180, 5] # Os valores de limiar usados na detecção de borda da tatuagem

S = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # Elemento estruturante da dilatação da borda da tatuagem

filter_w_dim = (3,3) # Tamanho inicial da janela quadrada da correlação 2d que será aplicada nnos pixels da tatuagem

# Função que aplica a correlação 2D em apenas um pixel especificado de uma imagem
def filter2D_pixelwise(image, kernel, coords):

    # Garante que o kernel seja float
    kernel = np.array(kernel, dtype=np.float32)

    # Pega as dimensões do kernel
    k_height, k_width = kernel.shape
    pad_y = k_height // 2
    pad_x = k_width // 2

    # Determina o pixel que vai sofrer a transformação
    y, x = coords

    # Extrai a vizinhança
    region = image[
        y - pad_y : y + pad_y + 1,
        x - pad_x : x + pad_x + 1
        ]

    # Multiplica pelo kernel e soma para cada camada de cor
    transformed_pixel = np.zeros_like(image[y,x])
    for i in range(3):
        transformed_pixel[i] = np.sum(region[:,:,i] * kernel)

    return transformed_pixel

# Função que cria um kernel de blur normal se baseando em uma região de uma máscara
def mask_normal_kernel(mask, dimension, coords):

    # Pega dimensões do kernel
    k_height, k_width = dimension
    pad_y = k_height // 2
    pad_x = k_width // 2

    # Determina a posição na máscara que vai determinar o kernel
    y, x = coords

    # O kernel é determinado a partir da região determinada da máscara
    kernel = mask[
        y - pad_y : y + pad_y + 1,
        x - pad_x : x + pad_x + 1
        ]

    # Converte kernel para float32
    kernel = np.float32(kernel)

    # Caso todos os pixels da região determinada da máscara forem iguais a 0
    # (Ou seja, todos os pixels da janela são de tatuagem e não pele)
    # O kernel é recalculado com uma janela maior 
    if (np.sum(kernel) == 0):
        kernel = mask_normal_kernel(mask, (k_height+2,k_width+2), coords)

    # Caso tenha pixels válidos na janela (pixels de pele)
    else:
        # Normaliza o kernel pra soma = 1
        kernel /= np.sum(kernel)

    return kernel

# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/Tatoo1.jpg')

# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# ----------------------------------------------------------
# Algoritmo de eliminação da tatuagem

# Isolamento da região de interesse que se deseja processar
I_output = I_input.copy()
I_tattoo = I_output[tattoo_region[0][0]:tattoo_region[1][0], tattoo_region[0][1]:tattoo_region[1][1]]

# Detecção de bordas mais dilatação das bordas
I_border = cv2.Canny(I_tattoo, border_tau[0], border_tau[1])
I_border = cv2.morphologyEx(I_border, cv2.MORPH_DILATE, S)
I_border = cv2.morphologyEx(I_border, cv2.MORPH_CLOSE, S)

# Criação da máscara que mostra onde há pele
# Que é o inverso da máscara de tatuagem
I_skin = cv2.bitwise_not(I_border)

# Aplicação da correlação nas regiões de borda da tatuagem
I_tattoo_lines, I_tattoo_columns, _ = I_tattoo.shape

# Percorre a região de interesse e verifica se cada pixel faz parte da borda
for y in np.arange(0, I_tattoo_lines):
    for x in np.arange(0, I_tattoo_columns):
        if I_border[y,x]:
            # Se o pixel fizer parte da borda, é criado um kernel que leva a posição do pixel atual como parâmetro
            kernel = mask_normal_kernel(I_skin, filter_w_dim, (y,x))
            # Após isso, aplica a correlação 2D no pixel atual de borda com o kernel recém criado
            I_tattoo[y,x] = filter2D_pixelwise(I_tattoo, kernel, (y,x))

# Exibição das imagens
ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem original')

# ax2.imshow(I_border, cmap='gray')
# ax2.set_title('Imagem de borda')

ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagem suavizada')

plt.show()