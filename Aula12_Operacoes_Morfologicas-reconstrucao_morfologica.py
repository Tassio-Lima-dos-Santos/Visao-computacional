import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

def image_reconstruction(I_mascara, I_marcador, S):
    num_pixel = 0

    while num_pixel != np.sum(I_marcador):
        num_pixel = np.sum(I_marcador)
        I_marcador = cv2.dilate(I_marcador, S)
        I_marcador = cv2.bitwise_and(I_marcador, I_mascara)

    return I_marcador

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# Limiarização Global

limiar = 128
ret, I2 = cv2.threshold(I, limiar, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(I2, cmap='gray')

# Reconstrução Morfológica

I_marcador = np.zeros(I2.shape, np.uint8)
I_marcador[637, 336] = 255

I_mascara = I2

S = np.ones((3,3), np.float32)
I_final = image_reconstruction(I_mascara, I_marcador, S)

plt.figure()
plt.imshow(I_final, cmap='gray')

plt.show()