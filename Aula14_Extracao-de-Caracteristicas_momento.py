import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

def bounding_box(I_component):
    y, x = np.where(I_component)

    ymin = np.min(y)
    xmin = np.min(x)
    ymax = np.max(y)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return (p0, p1)

def mpq(I, p, q): 
    y, x = np.where(I)

    mpq = np.sum((x ** p) * (y ** q))

    return mpq

def centroide(I):
    # calculo dos momentos
    m00 = mpq(I, 0, 0)
    m10 = mpq(I, 1, 0)
    m01 = mpq(I, 0, 1)

    # centroide
    xc = m10/m00
    yc = m01/m00

    c0 = np.array([xc, yc])

    return c0

# Parâmetros
component = 2

I = cv2.imread('./Imagens/formas.png', cv2.IMREAD_GRAYSCALE)
# plt.figure()
# plt.imshow(I, cmap='gray')

num_elements , I_labels = cv2.connectedComponents(I)

I_component = np.zeros((num_elements, I_labels.shape[0], I_labels.shape[1]), np.uint8)

for desired_shape in np.arange(0, num_elements):
    I_component[desired_shape] = np.uint8(I_labels == desired_shape) * 255

p0, p1 = bounding_box(I_component[component])

I2 = cv2.cvtColor(I_component[component], cv2.COLOR_GRAY2BGR)

color = (255, 128, 128)
thickness = 1
cv2.rectangle(I2, p0, p1, color, thickness)
# plt.figure()
# plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

c0 = centroide(I_component[component])

#desenha centroide
color = (255, 128, 128)
thickness = -1
radius = 2

cv2.circle(I2, np.int32(c0), radius, color, thickness)
plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

plt.show()
