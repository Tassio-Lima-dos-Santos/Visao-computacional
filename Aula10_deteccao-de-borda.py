import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
# plt.figure()
# plt.imshow(I1, cmap='gray')
# cv2.imshow('Imagem original', I1)

# Kernel da derivada parcial de x
Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
print(Kx)
Ix = cv2.filter2D(I1, cv2.CV_32F, Kx, borderType=cv2.BORDER_REFLECT101)
# print(Ix)
# plt.figure()
# plt.imshow(Ix, cmap='gray')
# cv2.imshow('Imagem derivada parcial de x', Ix)

# Kernel da derivada parcial de y
Ky = np.transpose(Kx)
print(Ky)
Iy = cv2.filter2D(I1, cv2.CV_32F, Ky, borderType=cv2.BORDER_REFLECT101)
# print(Iy)
# plt.figure()
# plt.imshow(Iy, cmap='gray')
# cv2.imshow('Imagem derivada parcial de y', Iy)

# Imagem de magnitude do gradiente
M = np.sqrt(Ix**2 + Iy**2)
plt.figure()
plt.imshow(M, cmap='gray')
print(np.max(M))

# limiarização global
limiar = 230
_, Iborda = cv2.threshold(M, limiar, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(Iborda, cmap='gray')
plt.show()
# cv2.waitKey(0)