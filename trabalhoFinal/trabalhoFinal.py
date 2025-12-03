import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros do algoritmo
imagem_de_entrada = './trabalhoFinal/banco_de_imagens/nível 1/placa1.jpg'

# Canny
t_upper = 600
t_lower = 100

# Transformação morfológica
structElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Filtragem de contornos
minArea = 1000

# Inicialização das imagens
I_input = cv2.imread(imagem_de_entrada)
I_output = I_input.copy()
I_reference = cv2.imread('./trabalhoFinal/banco_de_imagens/nível 1/placa1.jpg')
I_template = cv2.imread('./trabalhoFinal/banco_de_imagens/fonte_mercosul.png')

# Tamanho da imagem da placa tratada
# W, H = (400,200)

def sort_points(pts):
    # pts: array (4,1,2)
    pts = pts.reshape(4, 2)

    # ordena por soma (x+y) e diferença (x-y)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)

    p0 = pts[np.argmin(soma)]   # top-left
    p2 = pts[np.argmax(soma)]   # bottom-right
    p1 = pts[np.argmin(diff)]   # top-right
    p3 = pts[np.argmax(diff)]   # bottom-left

    return np.array([p0, p1, p2, p3], dtype="float32")

def tamanho_da_placa(pts):
    (p0, p1, p2, p3) = pts

    largura1 = np.linalg.norm(p1 - p0)
    largura2 = np.linalg.norm(p2 - p3)
    largura = max(int(largura1), int(largura2))

    altura1 = np.linalg.norm(p3 - p0)
    altura2 = np.linalg.norm(p2 - p1)
    altura = max(int(altura1), int(altura2))

    return largura, altura

# Ajuste de perspectiva usando detecção de bordas e características de contorno

# Aumento de nitidez

# # Filtragem 2D pura
# k_sharp = np.array([
#     [ 0, -1,  0],
#     [-1,  5, -1],
#     [ 0, -1,  0]
# ])
# I_sharp = cv2.filter2D(I_input, -1, k_sharp)

# Unsharp mask
I_blur = cv2.GaussianBlur(I_input, (0,0), sigmaX=5)
I_sharp = cv2.addWeighted(I_input, 1.8, I_blur, -0.8, 0)

I_output = I_sharp

# Detecção de bordas com Canny
I_border = cv2.Canny(I_sharp, t_upper, t_lower)

# Fechamento morfológico para melhorar a qualidade dos contornos
I_border = cv2.morphologyEx(I_border, cv2.MORPH_CLOSE, structElem)

# Aquisição dos contornos
contours, _ = cv2.findContours(I_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []

for cnt in contours:
    # Filtro de área mínima
    area = cv2.contourArea(cnt)
    if area < minArea:
        continue

    # Aproximação poligonal (para achar quadrilátero)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    # Verifica se é quadrilátero
    if len(approx) != 4:
        continue    

    # Retangularidade comparando a área à área do menor retângulo possível
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    retangularidade = area / cv2.contourArea(box)
    if retangularidade < 0.65:
        continue

    # Circularidade modificada
    # perimeter = cv2.arcLength(cnt, True)
    # circularidade = 4 * np.pi * area / (perimeter ** 2)
    # if circularidade < 0.10 or circularidade > 0.35:
    #     continue


    filtered_contours.append(approx)

for quad in filtered_contours:
    cv2.drawContours(I_output, [quad], -1, (0,255,0), 3)

pts_src = sort_points(filtered_contours[0])

W, H = tamanho_da_placa(pts_src)

pts_dst = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ], dtype="float32")

# matriz de homografia
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# aplica homografia planar
I_ajusted = cv2.warpPerspective(I_input, M, (W, H))

# OCR com Template Matching
source_gray = cv2.cvtColor(I_ajusted, cv2.COLOR_BGR2GRAY)
_, source_thresh = cv2.threshold(source_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
template_gray = cv2.cvtColor(I_template, cv2.COLOR_BGR2GRAY)
_, template_thresh = cv2.threshold(template_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# Criação do alfabeto
contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
alphabet = "0NA1OB2PC3DQ4RE5SF6TG7UH8VI9JWKXLYMZ"
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

# Leitura e tratamento da imagem lida
contours_source, _ = cv2.findContours(source_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

found_chars = []
for contour in contours_source:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w > 5 and h > 80 and w < 150:
        char_roi = source_thresh[y:y+h, x:x+w]
        found_chars.append({"box": (x, y, w, h), "roi": char_roi})

found_chars.sort(key=lambda c: (c["box"][1] // 50 * 50, c["box"][0]))

# Fazer o "Match" e Reconstruir o Texto
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

# Visualização dos Resultados 
I_output = I_ajusted.copy()
for char_info in found_chars:
    (x, y, w, h) = char_info["box"]
    is_colon_dot = (w < 15 and h < 20 and 0.5 < w/float(h) < 1.5)
    is_frame = (h/float(w) > 10)
    if is_colon_dot or is_frame:
        continue
    cv2.rectangle(I_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
# Exibição dos resultados
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.axis('off')

ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem Original')

ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagem Tratada')

plt.show()