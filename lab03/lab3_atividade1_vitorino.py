import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_e_processar_imagem(caminho_arquivo):
    """Carrega uma imagem, converte para cinza e binariza."""
    img_color = cv2.imread(caminho_arquivo)
    if img_color is None:
        raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem '{caminho_arquivo}'. Verifique o caminho.")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) 
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    return img_color, img_thresh

# --- 1. Extrair os Templates das Letras ---

caminho_template = './lab03/banco_de_imagens/template_letras.png'
_, template_thresh = carregar_e_processar_imagem(caminho_template)

contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVXZW"
templates = {}
contours_template = sorted(contours_template, key=lambda c: cv2.boundingRect(c)[0])

if len(contours_template) != len(alphabet):
    print(f"Aviso: Encontrados {len(contours_template)} contornos no template, mas o alfabeto tem {len(alphabet)} letras.")

for i, contour in enumerate(contours_template):
    if i >= len(alphabet):
        break
    (x, y, w, h) = cv2.boundingRect(contour)
    letter_roi = template_thresh[y:y+h, x:x+w]
    letter_roi = cv2.resize(letter_roi, (50, 50))
    templates[alphabet[i]] = letter_roi


# --- 2. Reconhecer os Caracteres na Imagem de Origem ---

# *** AQUI ESTÁ A ÚNICA MUDANÇA: O CAMINHO DA IMAGEM DE ORIGEM ***
caminho_origem = './lab03/banco_de_imagens/im3.png' # <--- MUDANÇA AQUI
source_img_color, source_thresh = carregar_e_processar_imagem(caminho_origem)

contours_source, _ = cv2.findContours(source_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    
    if is_colon_dot:
        pass # Ignora os dois-pontos
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
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.title("Caracteres Detectados (Dois-pontos ignorados)")
plt.axis('off')
plt.show()