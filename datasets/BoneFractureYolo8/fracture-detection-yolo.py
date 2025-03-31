from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Caminho do modelo treinado
model_path = r"C:\Users\est.pedrolucca\Projetos\CompuerVison\bonefracture-ai\datasets\BoneFractureYolo8\runs\detect\train5\weights\best.pt"

# Validar se o modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

# Carregar o modelo YOLO treinado
model = YOLO(model_path)

# Função de pré-processamento da imagem
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None

    # Converter para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalizar histograma (melhora contraste)
    equalized_image = cv2.equalizeHist(gray_image)

    # Aplicar filtro de suavização (reduz ruído)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    # Converter de volta para RGB (para compatibilidade com YOLO)
    processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

    return processed_image

# Função para detectar fraturas
def detect_fracture(image_path):
    if not os.path.exists(image_path):
        print(f"Imagem não encontrada: {image_path}")
        return

    processed_image = preprocess_image(image_path)
    if processed_image is None:
        print("Erro no pré-processamento da imagem.")
        return  

    # Salvar a imagem pré-processada temporariamente
    temp_path = "temp_img.jpg"
    cv2.imwrite(temp_path, processed_image)

    if not os.path.exists(temp_path):
        print("Erro ao salvar a imagem pré-processada.")
        return

    # Fazer a inferência com YOLO (ajustando o threshold para forçar detecção)
    results = model(temp_path, conf=0.1)  # Reduzindo a confiança mínima para aumentar as detecções

    # Processar os resultados
    detected = False
    for result in results:
        boxes = result.boxes  # Coordenadas das detecções
        class_ids = result.boxes.cls # IDs das classes detectadas
        
        for i, box in enumerate(boxes):
            class_id  = int(class_ids[i])
            class_name = model.names[class_id] if class_id in model.names else "Fratura detectada"
            
            detected = True
            show_image_with_boxes(temp_path, result, class_name)
    
    # Se não detectar nada, marcar "Sem fratura" na imagem
    if not detected:
        mark_no_fracture(temp_path)

# Função para exibir a imagem com as detecções
def show_image_with_boxes(image_path, result, class_name):
    image = cv2.imread(image_path)

    for i, box in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        
        # Desenhar a caixa verde na imagem
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Exibir o nome da classe ou "Fratura detectada" se a classe não for reconhecida
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Função para marcar "Sem fratura" na imagem
def mark_no_fracture(image_path):
    image = cv2.imread(image_path)

    # Adicionar texto "Sem fratura detectada"
    cv2.putText(image, "SEM FRATURA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # Exibir a imagem final
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Testar imagens individuais
detect_fracture("raiox/Distal-Radius-Fracture-in-Singapore.jpg")
detect_fracture("raiox/fratura-de-tibia-e-fibula.jpg")

