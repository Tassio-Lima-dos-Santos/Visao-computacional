import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/binary_dots.jpg', cv2.IMREAD_GRAYSCALE)
ret, I = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure()
plt.imshow(I, cmap='gray')

I_final = visco.imfill(I)
plt.figure()
plt.imshow(I_final, cmap='gray')

plt.show()