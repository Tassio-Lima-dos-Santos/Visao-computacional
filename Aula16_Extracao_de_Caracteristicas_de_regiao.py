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
        dados_do_componente['image'] = I_component

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

        # Matriz de inércia da região
        J = np.array([[u20, u11], [u11, u02]])

        # Matriz de inércia da elipse equivalente
        m00 = mpq(I_component, 0, 0)
        Je = 4/m00 * J

        avalores, avetores = np.linalg.eig(Je)

        # raios da elipse
        raios = -np.sort(-np.sqrt(avalores))
        menor_raio = raios[0]
        maior_raio = raios[1]

        print(raios)

        pos = np.argmax(avalores)

        vx = avetores[0, pos]
        vy = avetores[1, pos]

        orientacao = np.rad2deg(np.arctan2(vy, vx))

        dados_do_componente['razao_raios'] = menor_raio/maior_raio
        dados_do_componente['orientacao'] = orientacao

        # contorno
        contours, hierarchy = cv2.findContours(I_component,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        xc = contours[0][:,0,0]
        yc = contours[0][:,0,1]

        dados_do_componente['contour_x'] = xc
        dados_do_componente['contour_y'] = yc

        # perimetro
        perimetro = calculaPerimetro(xc, yc)
        dados_do_componente['perimetro'] = perimetro

        # circularidade
        m00 = mpq(I_component, 0, 0)
        circularidade = 4*np.pi*m00/(perimetro**2)
        dados_do_componente['circularidade'] = circularidade

        # curva de distância e ângulo
        curva_distancia = calculaCurvaDistancia(xc, yc, p0)
        curva_angulo = calculaCurvaAngulo(xc, yc, p0)

        dados_do_componente['curva_distancia'] = curva_distancia
        dados_do_componente['curva_angulo'] = curva_angulo

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

def calculaPerimetro(x, y):

    N = len(x)
    perimetro = np.sqrt((y[-1]-y[0])**2 + (x[-1]-x[0])**2)

    for n in np.arange(0, N-1):
        distancia = np.sqrt((y[n]-y[n+1])**2 + (x[n]-x[n+1])**2)
        perimetro = perimetro + distancia

    return perimetro

def calculaCurvaDistancia(xc, yc, centroide):

    N = len(xc)
    x0 = centroide[0]
    y0 = centroide[1]

    curva_distancia = np.zeros(N)

    for n in np.arange(0, N):
        curva_distancia[n] = np.sqrt((yc[n]-y0)**2 + (xc[n]-x0)**2)
    
    return curva_distancia

def calculaCurvaAngulo(xc, yc, centroide):
    
    N = len(xc)
    x0 = centroide[0]
    y0 = centroide[1]

    curva_angulo = np.zeros(N)

    for n in np.arange(0, N):
        curva_angulo[n] = np.arctan2((yc[n]-y0),(xc[n]-x0))
    
    return curva_angulo

def interp(y, numero_de_pontos_desejados):

    N = len(y)

    yp = np.interp(np.linspace(0,N-1,numero_de_pontos_desejados), np.arange(0, N), y)

    return yp

def computeMatch(y1, y2):

    # remoção de offset
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)

    # normalização das curvas de distâncias
    y1n = y1/np.sqrt(np.sum(y1**2))
    y2n = y2/np.sqrt(np.sum(y2**2))

    # correlação circular
    curva_correlacao = np.zeros(len(y1n))

    for k in np.arange(0, len(y1n)):
        curva_correlacao[k] = np.sum(np.roll(y1n, k) * y2n)
    
    max_correlacao = np.max(curva_correlacao)

    return max_correlacao, curva_correlacao

# Parâmetros
# rotulo = 1

# lê arquivo de imagem
I = cv2.imread('./Imagens/sharks.png', cv2.IMREAD_GRAYSCALE)
ret, I = cv2.threshold(I, 200, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title('Imagem de entrada')

infoRegioes = analisaRegioes(I)

Icomponete1 = infoRegioes[1]['image']
Icomponete2 = infoRegioes[2]['image']
plt.figure()
plt.imshow(cv2.cvtColor(Icomponete1, cv2.COLOR_BGR2RGB))
plt.title('Imagem do componente 1')
plt.figure()
plt.imshow(cv2.cvtColor(Icomponete2, cv2.COLOR_BGR2RGB))
plt.title('Imagem do componente 2')

# curvas de distância e ângulos dos componentes selecionados
curva_distancia1 = infoRegioes[1]['curva_distancia']
curva_distancia2 = infoRegioes[2]['curva_distancia']

# verifica similaridade entre as regiões
max_correlacao, curva_correlacao = computeMatch(curva_distancia1, curva_distancia2)
print('Máxima correlacao:', max_correlacao)

# apresenta curvas de correlacao
plt.figure()
plt.plot(curva_correlacao)
plt.title('Curva correlação')
plt.show()

# cv2.waitKey(0)
