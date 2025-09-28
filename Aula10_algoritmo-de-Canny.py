import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

t_lower = 100
t_upper = 600
Iborda = cv2.Canny(I1, t_upper, t_lower)

plt.figure()
plt.imshow(Iborda, cmap='gray')
plt.show()