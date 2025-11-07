import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

I1 = cv2.imread('./Imagens/marcadores.jpg')
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I1 = cv2.resize(I1, (0,0), fx=0.25, fy=0.25)

# plt.figure()
# plt.imshow(I1, cmap='gray')

# Converte do tipo uint8 para float32
I2 = np.float32(I1)/255

R = cv2.cornerHarris(I2, 2, 3, 0.06)

limiar = 0.25*R.max()
I3 = np.uint8(R > limiar)
y, x = np.where(I3)

print(f'Número de pontos de interesse = {len(x)}')

# print(y)
# print(x)

I4 = cv2.cvtColor(I1, cv2.COLOR_GRAY2BGR)

for i in np.arange(0, len(x)):
    cv2.circle(I4, (x[i], y[i]), 10, (0, 255, 0), 1)

plt.figure()
plt.imshow(cv2.cvtColor(I4, cv2.COLOR_BGR2RGB))

plt.show()
