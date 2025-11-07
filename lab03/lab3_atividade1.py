import cv2
from matplotlib import pyplot as plt
import numpy as np
# import visaoComputacional as visco

# Parâmetros do código
imagem_entrada = './lab03/banco_de_imagens/im5.png'
porcentagem_de_matches_usados = 1
RANSAC_confidence = 0.999
Limiar = 200

# ----------------------------------------------------------
# Carregamento das imagens
I_reference = cv2.imread('./lab03/banco_de_imagens/im1.png')
I_input = cv2.imread(imagem_entrada)

plt.figure()
plt.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))

plt.figure()
plt.imshow(cv2.cvtColor(I_reference, cv2.COLOR_BGR2RGB))

# Detecção dos pontos do canto da etiqueta

# Conversão das imagens coloridas para cinza
I_input_gray = cv2.cvtColor(I_input, cv2.COLOR_BGR2GRAY)
I_reference_gray = cv2.cvtColor(I_reference, cv2.COLOR_BGR2GRAY)

# Limiarização inversa para separar a etiqueta do fundo
_, I_input_thresh = cv2.threshold(I_input_gray, Limiar, 255, cv2.THRESH_BINARY_INV)
_, I_reference_thresh = cv2.threshold(I_reference_gray, Limiar, 255, cv2.THRESH_BINARY_INV)

# Segmentação da borda da etiqueta
input_contours, _ = cv2.findContours(I_input_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
reference_contours, _ = cv2.findContours(I_reference_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

input_contour = max(input_contours, key=cv2.contourArea)
reference_contour = max(reference_contours, key=cv2.contourArea)

epsilon_input = 0.02 * cv2.arcLength(input_contour, True)
epsilon_reference = 0.02 * cv2.arcLength(reference_contour, True)

approx_input = cv2.approxPolyDP(input_contour, epsilon_input, True)
approx_reference = cv2.approxPolyDP(reference_contour, epsilon_reference, True)

corners_input = approx_input.reshape(4, 2)
corners_reference = approx_reference.reshape(4, 2)

M, mask = cv2.findHomography(corners_input, corners_reference)

# # Converte as bordas do tipo uint8 para float32
# I_input_frame = np.float32(I_input_frame)/255
# I_reference_frame = np.float32(I_reference_frame)/255

# # Localiza os pontos de interesse usando o algoritmo de Shi-Tomazi
# corners_input = cv2.goodFeaturesToTrack(I_input_frame, 8, 0.001, 10)
# corners_reference = cv2.goodFeaturesToTrack(I_reference_frame, 10, 0.001, 10)

# # Localiza os pontos de interesse usando o algoritmo de Shi-Tomazi
# corners_input = cv2.goodFeaturesToTrack(I_input_thresh, 50, 0.01, 100)
# corners_reference = cv2.goodFeaturesToTrack(I_reference_thresh, 50, 0.01, 100)

# for corner in corners_input:
#     x = int(corner[0][0])
#     y = int(corner[0][1])
#     cv2.circle(I_input, (x,y), 10, (0,255,0), 2)

# for corner in corners_reference:
#     x = int(corner[0][0])
#     y = int(corner[0][1])
#     cv2.circle(I_reference, (x,y), 10, (0,255,0), 2)

# plt.figure()
# plt.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))

# plt.figure()
# plt.imshow(cv2.cvtColor(I_reference, cv2.COLOR_BGR2RGB))

# # Conversão dos corners para keypoints
# corners_input = np.int32(corners_input)
# corners_reference = np.int32(corners_reference)
# kp_input = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=20) for c in corners_input]
# kp_reference = [cv2.KeyPoint(x=float(c[0]), y=float(c[1]), size=20) for c in corners_reference]

# # Instância do detector BRISK
# brisk = cv2.BRISK_create()

# kp_input, desc_input = brisk.compute(I_input, kp_input)
# kp_reference, desc_reference = brisk.compute(I_reference, kp_reference)

# # Instancia o detector e descritor SIFT
# sift = cv2.SIFT_create()

# # Acha os pontos de interesses e seus descritores das imagens de referência e entrada
# kp_reference, desc_reference = sift.compute(I_reference, kp_reference)
# kp_input, desc_input = sift.compute(I_input, kp_input)

# # Cria objeto BFMatcher
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# # Executa o match entre os descritores da imagem de referência e de entrada
# matches = bf.match(desc_input, desc_reference)

# # Ordena os matches pela menor distância
# matches = sorted(matches, key = lambda x:x.distance)
# N = int(len(matches)*porcentagem_de_matches_usados)

# # Desenha matches
# I3 = cv2.drawMatches(I_input, kp_input, I_reference, kp_reference, matches[:N], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.figure()
# plt.imshow(cv2.cvtColor(I3, cv2.COLOR_BGR2RGB))

# # Obter matriz de homografia que mapeia a imagem de entrada na imagem de referência
# src_pts = np.zeros((N, 1, 2), np.float32)
# dst_pts = np.zeros((N, 1, 2), np.float32)

# for i, match in enumerate(matches[:N]):
#     src_pts[i] = np.float32(kp_input[match.queryIdx].pt).reshape(-1, 1, 2)
#     dst_pts[i] = np.float32(kp_reference[match.trainIdx].pt).reshape(-1, 1, 2)

# # Matriz de homografia
# M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC, confidence=RANSAC_confidence)

height, width, _ = I_reference.shape

I_ajusted_input = cv2.warpPerspective(I_input, M, (width, height))

plt.figure()
plt.imshow(cv2.cvtColor(I_ajusted_input, cv2.COLOR_BGR2RGB))

plt.show()
