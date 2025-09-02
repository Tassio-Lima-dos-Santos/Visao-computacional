import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

Limiar = 180

plt.figure()
plt.imshow(cv2.threshold(I, Limiar, 255, None, cv2.THRESH_BINARY)[1], cmap='gray')

# plt.figure()
# plt.imshow(visco.limiarizacao_global2(I, Limiar), cmap='gray')

plt.show()