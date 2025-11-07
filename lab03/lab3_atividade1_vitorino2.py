import cv2
import numpy as np
import matplotlib.pyplot as plt

def encontrar_maior_retangulo(imagem):
    # Converter para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas com Canny
    edged = cv2.Canny(blur, 50, 200)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Ordenar contornos por área em ordem decrescente
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Encontrar o contorno com a maior área que é aproximadamente um quadrilátero
    for contour in contours:
        # Aproximar o contorno
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Se tiver 4 pontos, assumimos que é o retângulo
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # Se não encontrou um quadrilátero, usar o maior contorno e obter o retângulo envolvente
    maior_contorno = contours[0]
    rect = cv2.minAreaRect(maior_contorno)
    box = cv2.boxPoints(rect)
    return box

def ordenar_pontos(pontos):
    # Ordenar os pontos: [superior esquerdo, superior direito, inferior direito, inferior esquerdo]
    # Soma das coordenadas: o ponto superior esquerdo tem a menor soma, o inferior direito tem a maior
    soma = pontos.sum(axis=1)
    pontos_ordenados = np.zeros((4, 2), dtype=np.float32)
    pontos_ordenados[0] = pontos[np.argmin(soma)]  # sup_esq
    pontos_ordenados[2] = pontos[np.argmax(soma)]  # inf_dir
    
    # Diferença: o ponto superior direito tem a menor diferença, o inferior esquerdo tem a maior
    diferenca = np.diff(pontos, axis=1)
    pontos_ordenados[1] = pontos[np.argmin(diferenca)]  # sup_dir
    pontos_ordenados[3] = pontos[np.argmax(diferenca)]  # inf_esq
    
    return pontos_ordenados

def corrigir_perspectiva(imagem, pontos):
    # Ordenar os pontos
    pts_origem = ordenar_pontos(pontos)
    
    # Calcular as dimensões do retângulo de destino
    largura_superior = np.linalg.norm(pts_origem[1] - pts_origem[0])
    largura_inferior = np.linalg.norm(pts_origem[2] - pts_origem[3])
    largura = max(int(largura_superior), int(largura_inferior))
    
    altura_esquerda = np.linalg.norm(pts_origem[3] - pts_origem[0])
    altura_direita = np.linalg.norm(pts_origem[2] - pts_origem[1])
    altura = max(int(altura_esquerda), int(altura_direita))
    
    # Definir os pontos de destino
    pts_destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype=np.float32)
    
    # Calcular a matriz de transformação e aplicar
    matriz = cv2.getPerspectiveTransform(pts_origem, pts_destino)
    imagem_corrigida = cv2.warpPerspective(imagem, matriz, (largura, altura))
    
    # Verificar a orientação: se a altura for maior que a largura, rotacionar 90 graus
    if imagem_corrigida.shape[0] > imagem_corrigida.shape[1]:
        imagem_corrigida = cv2.rotate(imagem_corrigida, cv2.ROTATE_90_CLOCKWISE)
    
    return imagem_corrigida

def processar_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro ao carregar {caminho_imagem}")
        return None
    
    pontos = encontrar_maior_retangulo(imagem)
    imagem_corrigida = corrigir_perspectiva(imagem, pontos)
    return imagem_corrigida

# Lista de imagens
imagens = [
    './lab03/banco_de_imagens/im4.png',
    './lab03/banco_de_imagens/im5.png',
    './lab03/banco_de_imagens/im6.png',
    './lab03/banco_de_imagens/im7.png',
    './lab03/banco_de_imagens/im8.png'
]

for caminho in imagens:
    corrigida = processar_imagem(caminho)
    if corrigida is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(corrigida, cv2.COLOR_BGR2RGB))
        plt.title(caminho)
        plt.axis('off')
        plt.show()