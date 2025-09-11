import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)
plt.figure
plt.imshow(I, cmap='gray')

n_linhas, n_colunas = I.shape

# Transformação de rotação
theta = 90
theta_rad = np.deg2rad(theta)
dx = n_linhas*np.sin(theta_rad)

A = np.array([[np.cos(theta_rad),-np.sin(theta_rad),dx], \
              [np.sin(theta_rad),np.cos(theta_rad),0]], \
              np.float32)

print(A)


n_linhas_final = int(n_colunas*np.sin(theta_rad) + n_linhas*np.cos(theta_rad))
n_colunas_final = int(n_linhas*np.sin(theta_rad) + n_colunas*np.cos(theta_rad))

I2 = cv2.warpAffine(I, A, (n_colunas_final, n_linhas_final))
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()