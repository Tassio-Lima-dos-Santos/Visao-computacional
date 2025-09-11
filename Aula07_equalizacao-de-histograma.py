import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/castle.jpg', cv2.IMREAD_GRAYSCALE)

# Histograma de I1

hist1 = cv2.calcHist([I1], [0], None, [256], [0,256])

# Equalização de Histograma

hist1 = hist1[:,0]

# pdf (função de densidade de probabilidade)

pdf = hist1/(I1.size)

# função de distribuição acumulada

cdf = np.cumsum(pdf)

# função de processamento/mapeamento 

f = np.uint8(255*cdf)

# Equalização de histograma

n_linhas, n_colunas = I1.shape
I2 = np.zeros((n_linhas, n_colunas), np.uint8)

for x in np.arange(0, n_colunas):
    for y in np.arange(0, n_linhas):
        I2[y, x] = f[I1[y, x]]

# Histograma de I2

hist2 = cv2.calcHist([I2], [0], None, [256], [0,256])
hist2 = hist2[:,0]

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0,0].imshow(I1, cmap='gray')
axs[0,1].bar(np.arange(0,256), hist1)
axs[1,0].imshow(I2, cmap='gray')
axs[1,1].bar(np.arange(0,256), hist2)
plt.show()