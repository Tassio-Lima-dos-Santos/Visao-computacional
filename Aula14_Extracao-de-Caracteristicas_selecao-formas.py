import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

I = cv2.imread('./Imagens/formas.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

num_elements , I_labels = cv2.connectedComponents(I)

desired_shape = 2
I_labels = np.uint8(I_labels == desired_shape) * 255
plt.figure()
plt.imshow(I_labels, cmap='gray')

plt.show()
