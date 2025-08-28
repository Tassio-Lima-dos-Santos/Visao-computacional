import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/quadrados_cinzas.png', cv2.IMREAD_GRAYSCALE)

print(I.shape)
print(I)

print(I[100, 100])
print(I[100, 300])
print(I[300, 100])
print(I[300, 300])

plt.figure()
plt.imshow(I, cmap='gray')
plt.show()