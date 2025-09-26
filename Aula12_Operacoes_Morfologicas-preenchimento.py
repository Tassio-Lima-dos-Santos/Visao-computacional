import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/binary_dots.jpg', cv2.IMREAD_GRAYSCALE)
ret, I = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.figure()
# plt.imshow(I, cmap='gray')

# Inversão da imagem binária
I_mascara = 255 - I
plt.figure()
plt.imshow(I_mascara, cmap='gray')

I_marcador = I_mascara.copy()
I_marcador[2:-1,2:-1] = 0
plt.figure()
plt.imshow(I_marcador, cmap='gray')

I_reconstruida = visco.imreconstruction(I_mascara, I_marcador)
plt.figure()
plt.imshow(I_reconstruida, cmap='gray')

I_final = 255 - I_reconstruida
plt.figure()
plt.imshow(I_final, cmap='gray')

plt.show()