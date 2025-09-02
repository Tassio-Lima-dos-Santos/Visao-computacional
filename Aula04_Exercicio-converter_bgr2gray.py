import cv2
from matplotlib import pyplot as plt
import numpy as np

I_bgr = cv2.imread('./Imagens/flowers4.png')
cv2.imshow('Imagem BGR', I_bgr)
# cv2.waitKey(0)

B = I_bgr[:, :, 0]
G = I_bgr[:, :, 1]
R = I_bgr[:, :, 2]

I_cinza1 = np.uint8( (1/3)*B + (1/3)*G + (1/3)*R )
cv2.imshow('Imagem Cinza 1', I_cinza1)

I_cinza2 = np.uint8( 0.1140*B + 0.5870*G + 0.2989*R )
cv2.imshow('Imagem Cinza 2', I_cinza2)

I_cinza3 = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem Cinza 3', I_cinza3)

cv2.waitKey(0)