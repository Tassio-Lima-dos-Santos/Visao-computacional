import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

hist = np.zeros(256)

n_linhas, n_colunas = I.shape

for x in np.arange(0, n_colunas):
    for y in np.arange(0, n_linhas):
        hist[I[y, x]] += 1

plt.figure()
plt.bar(np.arange(0,256), hist)
plt.xlim([0,256])
plt.title('Histograma')
plt.ylabel('Contagem dos pixels')
plt.xlabel('Valores do pixels')

plt.show()