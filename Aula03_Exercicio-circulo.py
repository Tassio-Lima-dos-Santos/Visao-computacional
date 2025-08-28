import cv2
from matplotlib import pyplot as plt
import numpy as np

N, M = (100, 100)

I = np.zeros((N, M), np.uint8)

R = 30
x0, y0 = (50, 50)

for theta in np.arange(0, 360):
    xc = int(x0 + R*np.cos(np.deg2rad(theta)))
    yc = int(y0 + R*np.sin(np.deg2rad(theta)))

    I[y0:yc, x0:xc] = 255
    I[y0:yc, xc:x0] = 255
    I[yc:y0, x0:xc] = 255
    I[yc:y0, xc:x0] = 255


cv2.imshow('Imagem binaria', I)
cv2.waitKey(0)