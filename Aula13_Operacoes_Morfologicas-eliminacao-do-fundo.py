import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/rice.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# Limiarização global
# limiar = 128
# _, I_threshold = cv2.threshold(I, limiar, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.figure()
# plt.imshow(I_threshold, cmap='gray')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
# print(kernel)

# I2 = cv2.morphologyEx(I, cv2.MORPH_ERODE, kernel)
# plt.figure()
# plt.imshow(I2, cmap='gray')

# I3 = cv2.morphologyEx(I2, cv2.MORPH_DILATE, kernel)
# plt.figure()
# plt.imshow(I3, cmap='gray')

I2 = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
# plt.figure()
# plt.imshow(I2, cmap='gray')

I3 = I - I2
# plt.figure()
# plt.imshow(I3, cmap='gray')

limiar = 128
_, I_threshold = cv2.threshold(I3, limiar, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plt.figure()
# plt.imshow(I_threshold, cmap='gray')

I_graos_validos = visco.imclearboard(I_threshold)
plt.figure()
plt.imshow(I_graos_validos, cmap='gray')

plt.show()
