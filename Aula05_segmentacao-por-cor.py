import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/tomato_124.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

# segmentação por cor

r_ref = 166
g_ref = 32
b_ref = 33

delta = 20

R = I[:, :, 0]
G = I[:, :, 1]
B = I[:, :, 2]

Mb = (B >= b_ref - delta) & (B <= b_ref + delta)
Mg = (G >= g_ref - delta) & (G <= g_ref + delta)
Mr = (R >= r_ref - delta) & (R <= r_ref + delta)

M = Mb & Mg & Mr

n_linhas, n_colunas, n_camadas = I.shape
I_bin = np.zeros((n_linhas, n_colunas), np.uint8)
I_bin[M] = 255
# I_bin = M*255

plt.figure()
plt.imshow(I_bin, cmap='gray')

plt.show()