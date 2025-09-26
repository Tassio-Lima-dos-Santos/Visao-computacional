import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/rice.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# limiarização global
limiar = 128
ret, I2 = cv2.threshold(I, limiar, 255, cv2.THRESH_BINARY)

# plt.figure()
# plt.imshow(I2, cmap='gray')

# Operação de erosão

S = np.ones((2,2), np.uint8)
I3 = cv2.erode(I2, S, iterations=1)

plt.figure()
plt.imshow(I3, cmap='gray')

plt.show()