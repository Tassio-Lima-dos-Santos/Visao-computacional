import cv2
from matplotlib import pyplot as plt
import numpy as np

N, M = (400, 200)

I = np.zeros((M, N), np.uint8)

for x in np.arange(0, N):
    c = np.uint8( x*255/(N-1) )

    I[0:M-1, x] = c 

plt.figure()
plt.imshow(I, cmap='gray')
plt.show()