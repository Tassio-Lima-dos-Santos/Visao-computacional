import cv2
import numpy as np

# --- Abrir arquivos de vídeo ---
cap1 = cv2.VideoCapture('lab01/Chromakey.mp4')
cap2 = cv2.VideoCapture('lab01/Clouds.mp4')

if not cap1.isOpened() or not cap2.isOpened():
    raise IOError("Erro ao abrir os vídeos. Verifique caminhos e nomes.")

n_linhas  = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_colunas = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
fps       = cap1.get(cv2.CAP_PROP_FPS) or 24  # usa fps do vídeo1 ou 24 se não disponível

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video  = cv2.VideoWriter('resultado_atividade2.avi', fourcc, fps,
                         (n_colunas, n_linhas))

# --- Parâmetros do chroma key (HSV) ---
ref_color = np.array([82, 156, 223])   # H,S,V
delta     = np.array([8, 30, 40])      # tolerância

while True:
    ret1, I1 = cap1.read()
    ret2, I2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Redimensiona o segundo vídeo para mesmo tamanho
    if I2.shape[:2] != (n_linhas, n_colunas):
        I2 = cv2.resize(I2, (n_colunas, n_linhas))

    # --- Chroma key ---
    I_HSV = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)

    lower = np.clip(ref_color - delta, 0, 255)
    upper = np.clip(ref_color + delta, 0, 255)

    background_mask = cv2.inRange(I_HSV, lower, upper)
    front_mask      = cv2.bitwise_not(background_mask)

    I_clouds = cv2.bitwise_and(I2, I2, mask=background_mask)
    I_front  = cv2.bitwise_and(I1, I1, mask=front_mask)
    I_final  = cv2.add(I_front, I_clouds)  # soma segura

    video.write(I_final)

    # Exibição opcional
    cv2.imshow("Chroma Key", I_final)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Liberação de recursos ---
cap1.release()
cap2.release()
video.release()
cv2.destroyAllWindows()