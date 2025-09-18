import cv2
from matplotlib import pyplot as plt
import numpy as np

I = np.zeros((5, 8), np.uint8)
I[:, 5:] = 255
print(I)

# Kernel
K = np.ones((3, 3), np.float32)
print(K)

# Filtragem 2D

O = cv2.filter2D(I, cv2.CV_32F, K, borderType=cv2.BORDER_REFLECT101)
print(O)