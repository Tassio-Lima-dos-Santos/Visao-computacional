import cv2
from matplotlib import pyplot as plt
import numpy as np

I = cv2.imread('./Imagens/tomato_124.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

# segmentação por cor

cor_ref = np.array([33, 32, 166])
delta = 20

limiar_superior = cor_ref + delta
limiar_inferior = cor_ref - delta

Ibin = cv2.inRange(I, limiar_inferior, limiar_superior)

plt.figure()
plt.imshow(Ibin, cmap='gray')
plt.show()