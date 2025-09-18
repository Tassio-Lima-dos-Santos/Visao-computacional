import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

# Kernel
K = (np.ones((3, 3), np.float32))
print(K)

# Filtragem 2D
I2 = cv2.filter2D(I, -1, K, borderType=cv2.BORDER_REFLECT101)

plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()