import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/quadrados.png')
print(I1.dtype)
cv2.imshow('Imagem I1', I1)

I2 = np.float32(I1/255)
print(I2.dtype)
cv2.imshow('Imagem I2', I2)

I3 = np.uint8(I2*255)
print(I3.dtype)
cv2.imshow('Imagem I3', I3)

cv2.waitKey(0)