import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

I = cv2.imread('./Imagens/formas.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

num_elements , I_labels = cv2.connectedComponents(I)

print(num_elements)
print(I_labels.shape)
print(np.max(I_labels))
print(np.min(I_labels))

plt.show()
