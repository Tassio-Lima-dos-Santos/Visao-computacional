import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/monalisa2.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# Kernel gaussiano
w = 5
sigma = (w - 1)/6
K = visco.gaussianKernel(w, sigma)
print(K)

# Filtragem 2D
I2 = cv2.filter2D(I, -1, K, borderType=cv2.BORDER_REFLECT101)

plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()