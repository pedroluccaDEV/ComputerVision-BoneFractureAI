from ultralytics import YOLO

# Carregar um modelo YOLO pré-treinado (pode usar um modelo YOLOv8 pré-treinado)
model = YOLO("yolov8s.pt")  # Carrega o modelo YOLOv8 pequeno. Você pode escolher outro tamanho se preferir.

# Agora, treinamos o modelo com o dataset.yaml fornecido.
model.train(
    data="data.yaml",  # Caminho correto para o arquivo YAML com as informações do dataset
    epochs=10,            # Número de épocas de treinamento (ajustar conforme necessário)
    imgsz=640,            # Tamanho das imagens de entrada
    batch=16,             # Tamanho do lotes
    workers=4,            # Número de workers para carregar os dados
    device="cpu"          # Usando a CPU, se tiver uma GPU disponível, pode colocar "cuda"
)

# Salvar o modelo treinado
model.save("bone_fracture_yolo_model.pt")
print("✅ Modelo YOLOv8 treinado e salvo com sucesso!")
