import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)
plt.figure
plt.imshow(I, cmap='gray')

n_linhas, n_colunas = I.shape

# Transformação de espelhamento
ex = 1 # 0 = não espelhar; 1 = espelhar
ey = 1

dx = n_colunas*(ex)
dy = n_linhas*(ey)

A = np.array([[1 - (2*ex),0,dx], \
              [0,1 - (2*ey),dy]], \
              np.float32)

print(A)


n_linhas_final = n_linhas
n_colunas_final = n_colunas

I2 = cv2.warpAffine(I, A, (n_colunas_final, n_linhas_final))
plt.figure
plt.imshow(I2, cmap='gray')
plt.show()