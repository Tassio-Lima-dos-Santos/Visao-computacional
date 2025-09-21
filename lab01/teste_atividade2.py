import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
cap1 = cv2.VideoCapture('lab01/Chromakey.mp4')
cap2 = cv2.VideoCapture('lab01/Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

# ----------------------------------------------------------
# Gera figura
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=0.9)

fig_width = 20
fig_height = 6
fig.set_size_inches((fig_width/2.54, fig_height/2.54))
fig.tight_layout()

# ----------------------------------------------------------
ret = True

# Seção de variáveis

ref_color = np.array([82, 156, 223]) # Cor de referência do chromakey no formato HSV
delta = np.array([3, 16, 23]) # A variação entorno da cor de referência que ainda vai ser considerada chromakey

while ret:

    # lê frame dos vídeos
    ret, I1 = cap1.read()
    _, I2 = cap2.read()

    if I1 is None:
        break

    # Algoritmo de detecção de objetos

    # Conversão da imagem original para HSV
    I_HSV = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)

    # Criação de máscaras
    background_mask = cv2.inRange(I_HSV, ref_color - delta, ref_color + delta)
    front_mask = cv2.bitwise_not(background_mask)

    # Aplicação do vídeo de background na região de chromakey
    I_clouds = cv2.bitwise_and(I2, I2, mask=background_mask)

    # Remoção da região de chromakey da imagem original
    I_front = cv2.bitwise_and(I1, I1, mask=front_mask)

    # Junção do background de nuvens junto com a imagem original
    I_final = I_front+I_clouds

    # atualiza plot
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

    ax2.clear()
    ax2.imshow(background_mask, cmap='gray')
    # ax2.plot([n_colunas/2, n_colunas/2], [0, n_linhas-1], ':')

    '''ax3.clear()
    ax3.plot(np.arange(0, ref_linha.size), ref_linha)
    ax3.set_ylim([0, 260])
    ax3.set_xlim([0, ref_linha.size])
    ax3.set_title('Coluna central da\n imagem binária')'''
    plt.pause(0.001)

print('Processo finalizado!')
cv2.waitKey(0)