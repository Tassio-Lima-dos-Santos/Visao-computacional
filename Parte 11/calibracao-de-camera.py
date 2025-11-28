import cv2
import glob
import numpy as  np
from matplotlib import pyplot as plt
import visaoComputacional as visco

imagens = glob.glob('/home/tassio/UFSC/Visão Computacional em Robótica/Visao-computacional/Parte 11/Dataset4/*.JPG')

# Arrays para armazenar as coordenadas de cada corner
objpoints = [] # Lista que armazena as coordenadas de cada ponto no mundo real
imgpoints = [] # Lista que armazena as coordenadas de cada ponto na imagem

# coordenadas de cada ponto no mundo real
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 25

print(objp)

for file_name in imagens:

    I1 = cv2.imread(file_name)
    I2 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    # encontra os pontos de interesse (corners) no tabuleiro
    ret, corners = cv2.findChessboardCorners(I1, (9,6), None)

    if ret == True:

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(I2, corners, (11, 11), (-1, -1), criteria)
        
        imgpoints.append(corners2)
        objpoints.append(objp)

        '''print(corners2)

        # Apresenta resultado parcial
        I3 = I1.copy()
        cv2.drawChessboardCorners(I3, (9,6), corners2, ret)
        plt.figure()
        plt.imshow(cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))
        plt.show()'''

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, I2.shape[::-1], None, None)

print('Matriz de parâmetros intrísecos da câmera:\n')
print(mtx)

K = np.zeros((3,4), np.float32)
K[:, :3] = mtx
print(K)

print('\nCoeficientes de distorção da lente\n')
print(dist)

# Matriz de parâmetros extrínsecos
indice = 0
M = np.zeros((4,4), np.float32)

# Matriz de rotação
R, _ =  cv2.Rodrigues(rvecs[indice])
print('\nMatriz de rotação:\n')
print(R)

# Vetor de translação
vetor_translacao = tvecs[indice]
print('\nVetor de translação:\n')
print(vetor_translacao)

M[:3, :3] = R
M[:3, 3] = vetor_translacao[:, 0]
M[3, 3] = 1

print('\nMatriz de parâmetros extrínsecos da câmera:\n')
print(M)

# Matriz de projeção
P = np.matmul(K, M)

print('\nMatriz de projeção:\n')
print(P)