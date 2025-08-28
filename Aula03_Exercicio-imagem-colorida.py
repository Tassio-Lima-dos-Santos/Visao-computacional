import cv2
from matplotlib import pyplot as plt
import numpy as np

M, N = (200, 400)

# Imagem BGR
I = np.zeros((M, N, 3), np.uint8)

I[50:150, 20:380] = [0, 255, 255]

# for x in np.arange(20, 380):
#     for y in np.arange(50, 150):
#         I[y, x] = [0, 255, 255]

# cv2.imshow('Imagem colorida', I)
# cv2.waitKey(0)

plt.figure()
plt.imshow( cv2.cvtColor(I, cv2.COLOR_BGR2RGB) )
plt.show()