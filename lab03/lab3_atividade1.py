import cv2
from matplotlib import pyplot as plt
import numpy as np
import visaoComputacional as visco

# Parâmetros do código
imagem_entrada = './lab03/banco_de_imagens/im7.png'

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

        pos = np.argmax(avalores)

        vx = avetores[0, pos]
        vy = avetores[1, pos]

        orientacao = np.rad2deg(np.arctan2(vy, vx))

        dados_do_componente['razao_raios'] = menor_raio/maior_raio
        dados_do_componente['orientacao'] = orientacao

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

# ----------------------------------------------------------
# Carregamento das imagens
I_reference = cv2.imread('./lab03/banco_de_imagens/im1.png')
I_input = cv2.imread(imagem_entrada)

plt.figure()
plt.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))

# AJUSTE DA PERSPECTIVA

# Conversão das imagens coloridas para cinza
I_input_gray = cv2.cvtColor(I_input, cv2.COLOR_BGR2GRAY)
I_reference_gray = cv2.cvtColor(I_reference, cv2.COLOR_BGR2GRAY)

# Limiarização inversa para separar a etiqueta do fundo
_, I_input_thresh = cv2.threshold(I_input_gray, 200, 255, cv2.THRESH_BINARY_INV)
_, I_reference_thresh = cv2.threshold(I_reference_gray, 200, 255, cv2.THRESH_BINARY_INV)

# Análise das características de região
info_input = analisaRegioes(I_input_thresh)

# Ajuste do ângulo da etiqueta
n_linhas, n_colunas, _ = I_input.shape

# Orientar a imagem de forma que o ângulo da etiqueta seja 0.15 graus
theta = -info_input[0]['orientacao'] - 0.15
theta_rad = np.deg2rad(theta)
dx = n_linhas*np.sin(theta_rad)
A = np.array([[np.cos(theta_rad),-np.sin(theta_rad),dx], \
              [np.sin(theta_rad),np.cos(theta_rad),0]], \
              np.float32)
n_linhas_final = int(n_colunas*np.sin(theta_rad) + n_linhas*np.cos(theta_rad))
n_colunas_final = int(n_linhas*np.sin(theta_rad) + n_colunas*np.cos(theta_rad))
I_input = cv2.warpAffine(I_input, A, (n_colunas_final, n_linhas_final))

# Redimensionar a imagem para o tamanho original
x1, y1 = np.abs(int((n_colunas_final-n_colunas)/2)), np.abs(int((n_linhas_final-n_linhas)/2))
x2, y2 = int(n_colunas_final-x1), int(n_linhas_final-y1)
I_input = I_input[y1:y2, x1:x2]

# Ajuste de perspectiva usando o detector e descritor SIFT
# Instancia o detector e descritor SIFT
sift = cv2.SIFT_create()

# Acha os pontos de interesses e seus descritores das imagens de referência e entrada
kp_reference, desc_reference = sift.detectAndCompute(I_reference, None)
kp_input, desc_input = sift.detectAndCompute(I_input, None)

# Cria objeto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Executa o match entre os descritores da imagem de referência e de entrada
matches = bf.match(desc_input, desc_reference)

# Ordena os matches pela menor distância
matches = sorted(matches, key = lambda x:x.distance)
N = len(matches)

# Obter matriz de homografia que mapeia a imagem de entrada na imagem de referência
src_pts = np.zeros((N, 1, 2), np.float32)
dst_pts = np.zeros((N, 1, 2), np.float32)

for i, match in enumerate(matches):
    src_pts[i] = np.float32(kp_input[match.queryIdx].pt).reshape(-1, 1, 2)
    dst_pts[i] = np.float32(kp_reference[match.trainIdx].pt).reshape(-1, 1, 2)

# Matriz de homografia
M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

height, width, _ = I_reference.shape

I_ajusted_input = cv2.warpPerspective(I_input, M, (width, height))

# Leitura do texto
# --- 1. Extrair os Templates das Letras ---
def carregar_e_processar_imagem(caminho_arquivo):
    """Carrega uma imagem, converte para cinza e binariza."""
    img_color = cv2.imread(caminho_arquivo)
    if img_color is None:
        raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem '{caminho_arquivo}'. Verifique o caminho.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) 
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    return img_color, img_thresh

caminho_template = './lab03/banco_de_imagens/template_letras.png'
_, template_thresh = carregar_e_processar_imagem(caminho_template)

# Criação do alfabeto
contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVXZW"
templates = {}
contours_template = sorted(contours_template, key=lambda c: cv2.boundingRect(c)[0])

if len(contours_template) != len(alphabet):
    print(f"Aviso: Encontrados {len(contours_template)} contornos no template, mas o alfabeto tem {len(alphabet)} letras.")

# Isolamento da região de interesse das letras
for i, contour in enumerate(contours_template):
    if i >= len(alphabet):
        break
    (x, y, w, h) = cv2.boundingRect(contour)
    letter_roi = template_thresh[y:y+h, x:x+w]
    letter_roi = cv2.resize(letter_roi, (50, 50))
    templates[alphabet[i]] = letter_roi


# --- 2. Reconhecer os Caracteres na Imagem de Origem ---

# Leitura e tratamento da imagem lida
source_img_color = I_ajusted_input
source_img_gray = cv2.cvtColor(source_img_color, cv2.COLOR_BGR2GRAY) 
_, source_thresh = cv2.threshold(source_img_gray, 127, 255, cv2.THRESH_BINARY_INV)
source_thresh = visco.imclearboard(source_thresh)

contours_source, _ = cv2.findContours(source_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

found_chars = []
for contour in contours_source:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w > 5 and h > 10 and w < 150:
        char_roi = source_thresh[y:y+h, x:x+w]
        found_chars.append({"box": (x, y, w, h), "roi": char_roi})

found_chars.sort(key=lambda c: (c["box"][1] // 50 * 50, c["box"][0]))


# --- 3. Fazer o "Match" e Reconstruir o Texto ---

recognized_text = ""
last_box = None

for char_info in found_chars:
    box = char_info["box"]
    char_roi = char_info["roi"]
    x, y, w, h = box

    if last_box is not None:
        last_x, last_y, last_w, last_h = last_box
        
        if y > last_y + last_h:
            recognized_text += "\n"
        else:
            gap = x - (last_x + last_w)
            if gap > (last_h * 0.4):
                recognized_text += " "
    
    char_roi_resized = cv2.resize(char_roi, (50, 50))
    
    is_colon_dot = (w < 15 and h < 20 and 0.5 < w/float(h) < 1.5)
    is_frame = (h/float(w) > 10)
    
    if is_colon_dot or is_frame:
        continue # Ignora os dois-pontos e as barras do contornos
    else:
        scores = []
        for letter, template_roi in templates.items():
            result = cv2.matchTemplate(char_roi_resized, template_roi, cv2.TM_SQDIFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append((score, letter))
        
        best_match = sorted(scores, key=lambda s: s[0])[0]
        recognized_text += best_match[1]
    
    last_box = box

# --- Saída Final ---
print(recognized_text)

# --- 4. Visualização dos Resultados ---
img_with_boxes = source_img_color.copy()
for char_info in found_chars:
    (x, y, w, h) = char_info["box"]
    is_colon_dot = (w < 15 and h < 20 and 0.5 < w/float(h) < 1.5)
    is_frame = (h/float(w) > 10)
    if is_colon_dot or is_frame:
        continue
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.title("Caracteres Detectados (Dois-pontos ignorados)")
plt.axis('off')

plt.show()
