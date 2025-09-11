import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/capa.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

pts_src = np.array([[52,100], [396, 79], [83, 571], [448, 527]])

n_colunas = 250
n_linhas = 350

pts_dst = np.array([[0,0], [n_colunas, 0], [0, n_linhas], [n_colunas, n_linhas]])

H, _ = cv2.findHomography(pts_src, pts_dst)
print(H)

I2 = cv2.warpPerspective(I, H, (n_colunas, n_linhas))
plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
plt.show()