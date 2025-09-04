import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# limiarização
_, I_bin = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apresenta resultados

plt.figure()
plt.imshow(I_bin, cmap='gray')

plt.show()