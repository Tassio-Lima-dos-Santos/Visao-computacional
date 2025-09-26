import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ----------------------------------------------------------
# Abra o vídeo
cap = cv2.VideoCapture('lab01/sequencia1.mp4')

n_linhas = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ----------------------------------------------------------
# Criação das figuras
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=0.9)

fig.set_size_inches((20/2.54, 6/2.54))
fig.tight_layout()

# Parâmetros de segmentação
# Cor de referência (BGR) para amarelo – ajuste conforme necessário
b_ref, g_ref, r_ref = (40, 180, 220)   # exemplo aproximado
tau = 80                               # limiar de distância em intensidade

# Variáveis de contagem
obj_count = 0
prev_value = 0   # valor anterior da soma da coluna central

while True:
    ret, I1 = cap.read()
    if not ret:
        break

    # 1) Cálculo da imagem de distância
    B, G, R = cv2.split(I1.astype(np.float32))
    D = np.sqrt((B - b_ref)**2 + (G - g_ref)**2 + (R - r_ref)**2)

    # 2) Geração da imagem binária
    I2 = np.zeros_like(D, dtype=np.uint8)
    I2[D <= tau] = 255

    # 3) Extração da coluna central
    col_center = I2[:, n_colunas//2]
    soma_coluna = np.sum(col_center) / 255  # número de pixels brancos na coluna

    # 4) Detecção de novo objeto
    if soma_coluna > 10 and prev_value <= 10:
        obj_count += 1
    prev_value = soma_coluna

    # Atualização dos gráficos
    ax1.clear()
    ax1.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Frame original')

    ax2.clear()
    ax2.imshow(I2, cmap='gray')
    ax2.axvline(x=n_colunas/2, color='c', linestyle=':')
    ax2.set_title('Imagem binária')

    ax3.clear()
    ax3.plot(np.arange(0, n_linhas), col_center[::-1])  # plota a coluna central
    ax3.set_ylim([0, 260])
    ax3.set_xlim([0, n_linhas])
    ax3.set_title(f'Coluna central\nObjetos detectados: {obj_count}')

    plt.pause(0.05)

print(f'Processo finalizado! Total de bolas detectadas: {obj_count}')
cap.release()
cv2.destroyAllWindows()