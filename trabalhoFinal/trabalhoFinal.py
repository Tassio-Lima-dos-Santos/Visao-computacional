import cv2
import numpy as np
from matplotlib import pyplot as plt
import visaoComputacional as visco

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------PARÂMETROS--------------------------------------------------

imagem_de_entrada = './trabalhoFinal/banco_de_imagens/nível 2/placa14.jpg'
alphabet = "0NA1OB2PC3DQ4RE5SF6TG7UH8VI9JWKXLYMZ"

# Canny
t_upper = 600
t_lower = 100

# Transformação morfológica
structElemClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
structElemOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Filtragem de contornos
minArea = 1000

# Inicialização das imagens
I_input = cv2.imread(imagem_de_entrada)
I_reference = cv2.imread('./trabalhoFinal/banco_de_imagens/nível 1/placa3.jpg')
I_template = cv2.imread('./trabalhoFinal/banco_de_imagens/fonte_mercosul.png')

# ----------------------------------FIM DOS PARÂMETROS-------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# ----------------------------------ALGORITMO PRINCIPAL------------------------------------------------

def ALGORITMO_PRINCIPAL():
    

    # Ajuste de perspectiva usando detecção de bordas e características de contorno

    templates = criacaoAlfabeto(I_template, alphabet)

    I_sharp = aumentoNitidez(I_input, 5, 0.8)

    contours = aquisicaoContornos(I_sharp, t_upper, t_lower, structElemClose, structElemOpen)

    I_ajusted = ajustePerspectivaContornosRobusto(I_input, contours)
    if I_ajusted is not None:
        found_chars = identificacaoCaracteres(I_ajusted)
        recognized_text = templateMatching(found_chars, templates)
        if (len(recognized_text) == 7):
            I_output = marcarCaracteres(I_ajusted, found_chars)
            return (recognized_text, I_output)
        
        I_ajusted = ajustePerspectivaSift(I_input)
        found_chars = identificacaoCaracteres(I_ajusted)
        recognized_text = templateMatching(found_chars, templates)
        if (len(recognized_text) == 7):
            I_output = marcarCaracteres(I_ajusted, found_chars)
            return (recognized_text, I_output)
        

    I_ajusted = ajustePerspectivaContornos(I_input, contours)
    if I_ajusted is not None:
        found_chars = identificacaoCaracteres(I_ajusted)
        recognized_text = templateMatching(found_chars, templates)
        if (len(recognized_text) == 7):
            I_output = marcarCaracteres(I_ajusted, found_chars)
            return (recognized_text, I_output)
        
        I_ajusted = ajustePerspectivaSift(I_input)
        found_chars = identificacaoCaracteres(I_ajusted)
        recognized_text = templateMatching(found_chars, templates)
        if (len(recognized_text) == 7):
            I_output = marcarCaracteres(I_ajusted, found_chars)
            return (recognized_text, I_output)

    
    I_ajusted = ajustePerspectivaSift(I_input)
    found_chars = identificacaoCaracteres(I_ajusted)
    recognized_text = templateMatching(found_chars, templates)
    if (len(recognized_text) == 7):
        I_output = marcarCaracteres(I_ajusted, found_chars)
        return (recognized_text, I_output)
    else:
        return ('', I_input)

# ----------------------------------FIM DO ALGORITMO PRINCIPAL-----------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# ------------------------------------------FUNÇÕES----------------------------------------------------

def aumentoNitidez(I_entrada, sigma, weight):
    I_blur = cv2.GaussianBlur(I_entrada, (0,0), sigmaX=sigma)
    I_sharp = cv2.addWeighted(I_entrada, 1 + weight, I_blur, -weight, 0)

    return I_sharp

def aquisicaoContornos(I_entrada, t_upper, t_lower, structElemClose, structElemOpen):
    # Detecção de bordas com Canny
    I_border = cv2.Canny(I_entrada, t_upper, t_lower)

    # Fechamento morfológico para melhorar a qualidade dos contornos
    I_border = cv2.morphologyEx(I_border, cv2.MORPH_CLOSE, structElemClose)
    I_border = cv2.morphologyEx(I_border, cv2.MORPH_OPEN, structElemOpen)

    # Aquisição dos contornos
    contours, _ = cv2.findContours(I_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

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

def plate_shape(pts):
    (p0, p1, p2, p3) = pts

    largura1 = np.linalg.norm(p1 - p0)
    largura2 = np.linalg.norm(p2 - p3)
    largura = max(int(largura1), int(largura2))

    altura1 = np.linalg.norm(p3 - p0)
    altura2 = np.linalg.norm(p2 - p1)
    altura = max(int(altura1), int(altura2))

    return largura, altura

def ajustePerspectivaContornosRobusto(I_entrada, contornos):
    filtered_contours = []

    for cnt in contornos:
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
        perimeter = cv2.arcLength(cnt, True)
        circularidade = 4 * np.pi * area / (perimeter ** 2)
        if circularidade < 0.10 or circularidade > 0.35:
            continue


        filtered_contours.append(approx)

    if (filtered_contours == []):
        return None

    pts_src = sort_points(filtered_contours[0])

    W, H = plate_shape(pts_src)

    pts_dst = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype="float32")

    # matriz de homografia
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # aplica homografia planar
    I_ajusted = cv2.warpPerspective(I_entrada, M, (W, H))

    return I_ajusted

def ajustePerspectivaContornos(I_entrada, contornos):
    filtered_contours = []

    for cnt in contornos:
        # Filtro de área mínima
        area = cv2.contourArea(cnt)
        if area < minArea:
            continue

        # Retangularidade comparando a área à área do menor retângulo possível
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        retangularidade = area / cv2.contourArea(box)
        if retangularidade < 0.65:
            continue

        # Circularidade modificada
        perimeter = cv2.arcLength(cnt, True)
        circularidade = 4 * np.pi * area / (perimeter ** 2)
        if circularidade < 0.10 or circularidade > 0.35:
            continue


        filtered_contours.append(box)

    if (filtered_contours == []):
        return None

    pts_src = sort_points(filtered_contours[0])

    W, H = plate_shape(pts_src)

    pts_dst = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype="float32")

    # matriz de homografia
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # aplica homografia planar
    I_ajusted = cv2.warpPerspective(I_entrada, M, (W, H))

    return I_ajusted

def ajustePerspectivaSift(I_entrada):
    # Ajuste de perspectiva usando o detector e descritor SIFT
    # Instancia o detector e descritor SIFT
    sift = cv2.SIFT_create()

    # Acha os pontos de interesses e seus descritores das imagens de referência e entrada
    kp_reference, desc_reference = sift.detectAndCompute(I_reference, None)
    kp_input, desc_input = sift.detectAndCompute(I_entrada, None)

    # Cria objeto BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

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

    I_saida = cv2.warpPerspective(I_entrada, M, (width, height))

    return I_saida

def criacaoAlfabeto(I_entrada, alfabeto):
    template_gray = cv2.cvtColor(I_entrada, cv2.COLOR_BGR2GRAY)
    _, template_thresh = cv2.threshold(template_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Criação do alfabeto
    contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    templates = {}
    contours_template = sorted(contours_template, key=lambda c: cv2.boundingRect(c)[0])

    # Isolamento da região de interesse das letras
    for i, contour in enumerate(contours_template):
        if i >= len(alfabeto):
            break
        (x, y, w, h) = cv2.boundingRect(contour)
        letter_roi = template_thresh[y:y+h, x:x+w]
        letter_roi = cv2.resize(letter_roi, (50, 50))
        templates[alphabet[i]] = letter_roi

    return templates

def identificacaoCaracteres(I_entrada):
    source_gray = cv2.cvtColor(I_entrada, cv2.COLOR_BGR2GRAY)
    _, source_thresh = cv2.threshold(source_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # Remove os elementos na borda
    source_thresh = visco.imclearboard(source_thresh)

    # Leitura e tratamento da imagem lida
    contours_source, _ = cv2.findContours(source_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_chars = []
    for contour in contours_source:
        (x, y, w, h) = cv2.boundingRect(contour)
        razao_hw = h/float(w)
        invalide = (not(50 < w < 150) or not(100 < h < 250) or not(1.2 < razao_hw < 3))
        if not invalide:
            char_roi = source_thresh[y:y+h, x:x+w]
            found_chars.append({"box": (x, y, w, h), "roi": char_roi})

    found_chars.sort(key=lambda c: (c["box"][1] // 50 * 50, c["box"][0]))

    return found_chars

def templateMatching(found_chars, templates):
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
        
        razao_hw = h/float(w)
        invalide = (not(50 < w < 150) or not(100 < h < 250) or not(1.2 < razao_hw < 3))
        
        # invalide = False

        if invalide:
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

    return recognized_text

def marcarCaracteres(I_entrada, found_chars):
    # Visualização dos Resultados 
    I_saida = I_entrada.copy()
    for char_info in found_chars:
        (x, y, w, h) = char_info["box"]
        razao_hw = h/float(w)
        invalide = (not(50 < w < 150) or not(100 < h < 250) or not(1.2 < razao_hw < 3))

        # invalide = False

        if invalide:
            continue
        cv2.rectangle(I_saida, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return I_saida

# ----------------------------------FIM DAS FUNÇÕES----------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# ----------------------------------EXIBIÇÃO DOS RESULTADOS--------------------------------------------

(recognized_text, I_output) = ALGORITMO_PRINCIPAL()

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.axis('off')
ax2 = fig.add_subplot(1,2,2)
ax2.axis('off')

ax1.imshow(cv2.cvtColor(I_input, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagem Original')

# I_output = I_border
ax2.imshow(cv2.cvtColor(I_output, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagem Tratada')

# Adiciona texto na parte superior da figura inteira
if (len(recognized_text)):
    fig.text(0.5, 0.95, f"Placa identificada: {recognized_text}", ha='center', va='center', fontsize=14, fontweight='bold')

else:
    fig.text(0.5, 0.95, f"Placa não identificada", ha='center', va='center', fontsize=14, fontweight='bold')

plt.show()