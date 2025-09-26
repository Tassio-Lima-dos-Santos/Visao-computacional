import cv2
from matplotlib import pyplot as plt
import numpy as np

def similaridade(I1, I2):
    I1 = np.float32(I1)/255
    I2 = np.float32(I2)/255

    valor_similaridade = np.sum(np.abs(I1 - I2))

    return valor_similaridade

I = cv2.imread('./Imagens/Wally.png', cv2.IMREAD_GRAYSCALE)
T1 = cv2.imread('./Imagens/template2.png', cv2.IMREAD_GRAYSCALE)
# plt.figure()
# plt.imshow(I, cmap='gray')
# plt.figure()
# plt.imshow(T1, cmap='gray')

lin_I, col_I = I.shape
lin_T1, col_T1 = T1.shape

# Template Match

I_similaridade = cv2.matchTemplate(I, T1, cv2.TM_CCORR_NORMED)
lin_sim, col_sim = I_similaridade.shape

# Apresenta I_similaridade como uma superfície
y = np.arange(0, lin_sim)
x = np.arange(0, col_sim)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, I_similaridade, alpha=0.5, linewidth=0.5, edgecolors='k')

limiar = 0.97
D = I_similaridade >= limiar

y0, x0 = np.where(D)

I_final = I.copy()

for k in np.arange(0, x0.size):
    top_left = (x0[k], y0[k])
    bottom_right = (top_left[0] + col_T1, top_left[1] + lin_T1)
    cv2.rectangle(I_final, top_left, bottom_right, 0, 3)

plt.figure()
plt.imshow(I_final, cmap='gray')

plt.show()