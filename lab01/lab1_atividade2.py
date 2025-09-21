import visaoComputacional as visco
import cv2
import numpy as np

# abri arquivos de videos
cap1 = cv2.VideoCapture('lab01/Chromakey.mp4')
cap2 = cv2.VideoCapture('lab01/Clouds.mp4')

n_linhas = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('resultado_atividade2.avi', fourcc, 24, (n_colunas, n_linhas))

# Seção de variáveis

ref_color = np.array([82, 156, 223]) # Cor de referência do chromakey no formato HSV
delta = np.array([8, 30, 40]) # A variação entorno da cor de referência que ainda vai ser
                              # considerada como chromakey

# processa cada quadro dos vídeos
ret = True
while ret:

    # lê frame dos vídeos
    ret, I1 = cap1.read()
    _, I2 = cap2.read()

    if I1 is None:
        break

    # Efeito de chromakey
    
    # Conversão da imagem original para HSV
    I_HSV = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)

    # Criação de máscaras
    background_mask = cv2.inRange(I_HSV, ref_color - delta, ref_color + delta)
    front_mask = cv2.bitwise_not(background_mask)

    # Aplicação do vídeo de background na região de chromakey
    I_clouds = cv2.bitwise_and(I2, I2, mask=background_mask)

    # Remoção da região de chromakey da imagem original
    I_front = cv2.bitwise_and(I1, I1, mask=front_mask)

    # Junção do background de nuvens junto com a imagem original
    I_final = I_front+I_clouds

    video.write(I_final)

    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
video.release()