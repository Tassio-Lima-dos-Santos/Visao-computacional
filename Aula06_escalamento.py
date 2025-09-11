import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)
plt.figure
plt.imshow(I, cmap='gray')

# escalamento

sx = 10
sy = 2

I2 = cv2.resize(I, None, fx=sx, fy=sy)

# A = np.array([[sx, 0, 0], [0, sy, 0]], np.float32)

# n_linhas, n_colunas = I.shape
# n_linhas_final = int(sy*n_linhas)
# n_colunas_final = int(sx*n_colunas)


# I2 = cv2.warpAffine(I, A, (n_colunas_final, n_linhas_final))
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()