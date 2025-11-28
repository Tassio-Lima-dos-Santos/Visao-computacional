import cv2
import numpy as  np
from matplotlib import pyplot as plt

IL = cv2.imread('./Imagens/rochas-L.png', cv2.IMREAD_GRAYSCALE)
IR = cv2.imread('./Imagens/rochas-R.png', cv2.IMREAD_GRAYSCALE)

# plt.figure()
# plt.imshow(IL, cmap='gray')

# plt.figure()
# plt.imshow(IR, cmap='gray')

# Disparidade 
stereo = cv2.StereoBM.create(numDisparities=16*6, blockSize=15)
Idisp = stereo.compute(IR, IL)
Idisp = Idisp/16

# Profundidade
b = 0.16
Idisp = Idisp + 274
Z = 3740*b/Idisp

plt.figure()
plt.imshow(Z, cmap='gray')

# A = np.ones_like(Idisp) * np.min(Idisp)
# Idisp += A
# Idisp *= 255/np.max(Idisp)

# plt.figure()
# plt.imshow(Idisp, cmap='gray')

plt.show()