import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

I1 = cv2.imread('./Imagens/capa.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(I2, cmap = 'gray')

# Imagem de Borda
I3 = cv2.Canny(I2, 50, 150, apertureSize=3)
plt.figure()
plt.imshow(I3, cmap = 'gray')

# Extração de contornos
contours, hierarchy = cv2.findContours(I3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

I4 = np.zeros(I3.shape, np.uint8)
color = (255, 255, 255)

for i in range(len(contours)):
    if len(contours[i][:,0,0]) > 100:
        cv2.drawContours(I4, contours, i, color)

plt.figure()
plt.imshow(I4, cmap='gray')

plt.show()