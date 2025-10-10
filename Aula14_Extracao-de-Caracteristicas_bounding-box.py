import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

# Parâmetros
component = 2

def bounding_box(I_component):
    y, x = np.where(I_component)

    ymin = np.min(y)
    xmin = np.min(x)
    ymax = np.max(y)
    xmax = np.max(x)

    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])

    return (p0, p1)

I = cv2.imread('./Imagens/formas.png', cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(I, cmap='gray')

num_elements , I_labels = cv2.connectedComponents(I)

I_component = np.zeros((num_elements, I_labels.shape[0], I_labels.shape[1]), np.uint8)

for desired_shape in np.arange(0, num_elements):
    I_component[desired_shape] = np.uint8(I_labels == desired_shape) * 255

p0, p1 = bounding_box(I_component[component])

I2 = cv2.cvtColor(I_component[component], cv2.COLOR_GRAY2BGR)

color = (255, 128, 128)
thickness = 1
cv2.rectangle(I2, p0, p1, color, thickness)
plt.figure()
plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

plt.show()
