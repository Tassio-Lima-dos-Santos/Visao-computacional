import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros do algoritmo
imagem_de_entrada = './trabalhoFinal/banco_de_imagens/nível 1/placa2.jpg'

# Inicialização das imagens
I_input = cv2.imread(imagem_de_entrada)
I_reference = cv2.imread('./trabalhoFinal/banco_de_imagens/nível 1/placa1.jpg')
I_template = cv2.imread('./trabalhoFinal/banco_de_imagens/fonte_mercosul.png')

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

I_ajusted = cv2.warpPerspective(I_input, M, (width, height))

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