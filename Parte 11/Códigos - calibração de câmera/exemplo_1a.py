import numpy as np
import cv2
import glob
import visaoComputacional as visco

images = glob.glob('Aula 16\Dataset4\*.JPG')

# arrays para armazenar as coordenadas dos pontos do objeto (em relação ao
# sistema de coordenadas do espaço físico) e dos pontos da imagem (em relação
# ao sistema de coordendas da imagem)
objpoints = []
imgpoints = []

# prepara as coordenadas dos pontos do objeto, (0,0,0), (2.5, 0, 0), (5.0, 0, 0)
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 2.5

for fname in images:

    I1 = cv2.imread(fname)
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I1 = cv2.resize(I1, (0,0), fx=0.5, fy=0.5)

    ret, corners = cv2.findChessboardCorners(I1, (9,6), None)

    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(I1, corners, (11, 11), (-1,-1), criteria)

        imgpoints.append(corners2)
        objpoints.append(objp)

        '''cv2.drawChessboardCorners(I1, (9,6), corners2, ret)
        cv2.imshow(f'{fname}', I1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, I1.shape[::-1], None, None)

print('Parâmetros da lente:\n', dist)

print('Matriz de parâmetros intrísecos:\n', mtx)

indice_imagem = 1
print(images[indice_imagem])

# matriz de rotação
vetor_rotacao = rvecs[indice_imagem]
R, _ = cv2.Rodrigues(vetor_rotacao)
print('Matriz de rotação:\n', R)

# vetor de translação
vetor_translacao = tvecs[indice_imagem]
print('Vetor de translação:\n', vetor_translacao)

# matriz de projeção da câmera
K = np.zeros((3,4), np.float32)
K[0:3,0:3] = mtx

M = np.zeros((4,4), np.float32)
M[0:3,0:3] = R
M[0:3, 3] = -vetor_translacao[:,0]
M[3,3] = 1

P = np.linalg.matmul(K, M)

print('Matriz de projeção:\n', P)

P2 = np.delete(P, 2, 1)

print('Matriz de projeção reduzida:\n', P2)

# -------------------------------------------------------------
I2 = cv2.imread('Aula 16\Cenas\cena3.JPG')
I2 = cv2.resize(I2, (0,0), fx=0.5, fy=0.5)
cv2.imshow('Imagem da cena', I2)

# corrige distorções da lente
I3 = cv2.undistort(I2, mtx, dist, None)
cv2.imshow('Imagem da cena corrigida', I3)

# converte para escala de cinza
I4 = cv2.cvtColor(I3, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem da cena em escala de cinza', I4)

# realiza limiarização global
ret, I5 = cv2.threshold(I4, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Imagem da cena limiarizada', I5)

# 
I6 = visco.imclearboard(I5, 20)
cv2.imshow('Imagem da cena limiarizada sem regiões próximas a borda', I6)

#
infoRegioes = visco.analisaRegioes(I6)
print(f'Quantidade de regiões identificadas: {len(infoRegioes)}\n')

X = np.zeros(4, np.float32)
Y = np.zeros(4, np.float32)

for k in range(len(infoRegioes)):

    x = infoRegioes[k]['centroide']
    x = np.append(x, 1)
    x = x.reshape(3,1)
    
    Point = np.linalg.matmul( np.linalg.inv(P2), x )

    X[k] = Point[0,0]/Point[2,0]
    Y[k] = Point[1,0]/Point[2,0]

    print(f'Coordenadas do mundo real: ({X[k]}, {Y[k]}) cm')

for k in range(1, len(infoRegioes)):

    distancia = np.sqrt((X[0]-X[k])**2 + (Y[0]-Y[k])**2)
    print(f'Distância {distancia} cm\n')


cv2.waitKey(0)