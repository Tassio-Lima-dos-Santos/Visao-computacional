import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

def upq(I, p, q):

    c0 = centroide(I)
    xc, yc = c0

    y, x = np.where(I)

    momento_central = np.sum(((x-xc)**p) * ((y-yc)**q))

    return momento_central

def analisaRegioes(I):
    infoRegioes = []

    # rotulamento
    num_elem, I_labels = cv2.connectedComponents(I)

    for i in np.arange(1, num_elem):
        # dados do componente
        dados_do_componente = dict()

        # imagem do componente
        I_component = np.uint8(I_labels == i) * 255
        dados_do_componente['imagem'] = I_component

        # bounding box
        p0, p1 = bounding_box(I_component)
        dados_do_componente['bb_p0'] = p0
        dados_do_componente['bb_p1'] = p1

        # area
        m00 = mpq(I_component, 0, 0)
        dados_do_componente['area'] = m00

        # centroide 
        c0 = centroide(I_component)
        dados_do_componente['centroide'] = c0

        # momentos centrais de segunda ordem
        u20 = upq(I_component, 2, 0)
        u02 = upq(I_component, 0, 2)
        u11 = upq(I_component, 1, 1)

        # adiciona dicionario da lista infoRegioes
        infoRegioes.append(dados_do_componente.copy())

    return infoRegioes

def desenha_bounding_box(I, infoRegioes, color, thickness):
    num_regioes = len(infoRegioes)
    for i in np.arange(0, num_regioes):
        p0 = infoRegioes[i]['bb_p0']
        p1 = infoRegioes[i]['bb_p1']

        cv2.rectangle(I, p0, p1, color, thickness)

def desenha_centroide(I, infoRegioes, color, radius):
    num_regioes = len(infoRegioes)

    for i in np.arange(0, num_regioes):
        c0 = infoRegioes[i]['centroide']

        cv2.circle(I, np.int32(c0), radius, color, -1)

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
rotulo = 2

I = cv2.imread('./Imagens/formas.png', cv2.IMREAD_GRAYSCALE)
# plt.figure()
# plt.imshow(I, cmap='gray')

num_elem, I_labels = cv2.connectedComponents(I)

I2 = np.uint8(I_labels == rotulo) * 255
plt.figure()
plt.imshow(I2, cmap='gray')

# Momentos centrais
u20 = upq(I2, 2, 0)
u02 = upq(I2, 0, 2)
u11 = upq(I2, 1, 1)

print(f'u20 = {u20}\n')
print(f'u02 = {u02}\n')
print(f'u11 = {u11}\n')

plt.show()
