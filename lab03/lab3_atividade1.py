import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

# ----------------------------------------------------------
# Carregamento da imagem de entrada
I_input = cv2.imread('./lab02/castle.jpg', cv2.IMREAD_GRAYSCALE)