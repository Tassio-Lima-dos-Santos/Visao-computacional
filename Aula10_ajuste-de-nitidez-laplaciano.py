import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/moon2.jpg', cv2.IMREAD_GRAYSCALE)

# Kernel Laplaciano
Kl = np.array([[0,1,0], [1, -4, 1], [0,1,0]], np.float32)

# Filtragem com o laplaciano
Il = cv2.filter2D(I1, -1, Kl, borderType=cv2.BORDER_REFLECT101)

k = 0.2
In = I1 - k*Il
In[In > 1] = 1
In[In < 0] = 0

plt.figure()
plt.imshow(In, cmap='gray')
plt.show()