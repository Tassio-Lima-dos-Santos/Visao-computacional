import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/cell.png', cv2.IMREAD_GRAYSCALE)

_, Iotsu1 = cv2.threshold(I1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, Iotsu2 = cv2.threshold(I1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# plt.figure()
# plt.imshow(Iotsu1, cmap='gray')
# plt.figure()
# plt.imshow(Iotsu2, cmap='gray')

# Kernel da derivada parcial de x
Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
# print(Kx)
Ix = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
# Ixx = cv2.filter2D(Ix, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)

# Kernel da derivada parcial de x
Ky = np.transpose(Kx)
# print(Ky)
Iy = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
# Iyy = cv2.filter2D(Iy, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)

# Imagem de magnitude do gradiente
M = np.sqrt(Ix**2 + Iy**2)
plt.figure()
plt.imshow(M, cmap='gray')
print(np.max(M))

# limiarização global
limiar_borda = 100
_, Iborda = cv2.threshold(M, limiar_borda, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(Iborda, cmap='gray')
plt.show()
# cv2.waitKey(0)