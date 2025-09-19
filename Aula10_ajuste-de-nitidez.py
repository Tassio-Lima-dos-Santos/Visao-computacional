import cv2
from matplotlib import pyplot as plt
import numpy as np

I1 = cv2.imread('./Imagens/moon2.jpg', cv2.IMREAD_GRAYSCALE)
I1 = np.float32(I1)/255
plt.figure()
plt.imshow(I1, cmap='gray')

w = 11
sigma = (w-1)/6
Is = cv2.GaussianBlur(I1, (w,w), sigma)
# plt.figure()
# plt.imshow(Is, cmap='gray')

Ig = I1 - Is
# plt.figure()
# plt.imshow(Ig, cmap='gray')

k = 20
In = I1 + k*Ig
In[In > 1] = 1
In[In < 0] = 0
plt.figure()
plt.imshow(In, cmap='gray')

plt.show()