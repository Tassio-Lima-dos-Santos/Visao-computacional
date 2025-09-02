import cv2
from matplotlib import pyplot as plt
import numpy as np

I_bgr = cv2.imread('./Imagens/quadrados.png')

plt.figure()
plt.imshow(cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB))
plt.show()

I_hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)

print(f'I[100, 100] = {I_hsv[100, 100]}')
print(f'I[100, 300] = {I_hsv[100, 300]}')
print(f'I[300, 100] = {I_hsv[300, 100]}')
print(f'I[300, 300] = {I_hsv[300, 300]}')