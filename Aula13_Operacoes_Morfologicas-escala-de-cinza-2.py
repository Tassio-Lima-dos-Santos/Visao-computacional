import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/pcb.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print(kernel)

I2 = cv2.morphologyEx(I, cv2.MORPH_DILATE, kernel)
plt.figure()
plt.imshow(I2, cmap='gray')

plt.show()
