import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

I1 = cv2.imread('./Imagens/carta1.JPG')
I1 = cv2.resize(I1, (0, 0), fx=0.25, fy=0.25)

plt.figure()
plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(I2, cmap='gray')

# DETECTOR SIFT
sift = cv2.SIFT_create()
keypoints, descritores = sift.detectAndCompute(I2, None)

print(len(keypoints))
print(keypoints[0].pt)
print(keypoints[0].size)
print(keypoints[0].angle)

print(descritores.shape)
print(descritores[0,:])

# Desenha pontos/regiões de interesse localizadas
I3 = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)

cv2.drawKeypoints(I2, keypoints, I3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

plt.show()