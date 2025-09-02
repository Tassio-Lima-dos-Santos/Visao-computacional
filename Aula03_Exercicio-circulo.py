import cv2
from matplotlib import pyplot as plt
import numpy as np

N, M = (1000, 1000)

I = np.zeros((N, M), np.uint8)

R = 300
x0, y0 = (500, 500)

for r in np.arange(0, R):
    for theta in np.arange(0, 360):
        xc = int(x0 + (R-r)*np.cos(np.deg2rad(theta)))
        yc = int(y0 + (R-r)*np.sin(np.deg2rad(theta)))

        I[y0:yc, x0:xc] = 255*r/R
        I[y0:yc, xc:x0] = 255*r/R
        I[yc:y0, x0:xc] = 255*r/R
        I[yc:y0, xc:x0] = 255*r/R


plt.figure()
plt.imshow(I, cmap="gray")
plt.show()