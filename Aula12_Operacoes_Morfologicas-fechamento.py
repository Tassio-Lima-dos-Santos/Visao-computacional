import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

I = cv2.imread('./Imagens/tomato_124.jpg')
# plt.figure()
# plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

# Segmentação baseada em cor

color_ref = np.array([27,32,177])
limiar = 75
I2 = visco.color_segmentation(I, color_ref, limiar)

plt.figure()
plt.imshow(I2, cmap='gray')

# Operação morfológica de fechamento

S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,12))
# I3 = cv2.morphologyEx(I2, cv2.MORPH_DILATE, S)
# plt.figure()
# plt.imshow(I3, cmap='gray')

# I4 = cv2.morphologyEx(I3, cv2.MORPH_ERODE, S)
# plt.figure()
# plt.imshow(I4, cmap='gray')

I3 = cv2.morphologyEx(I2, cv2.MORPH_CLOSE, S)
plt.figure()
plt.imshow(I3, cmap='gray')

plt.show()