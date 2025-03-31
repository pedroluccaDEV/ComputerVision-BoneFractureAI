import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função de pré-processamento da imagem
def preprocess_image(image_path):
    # Carregar a imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Equalizar histograma para melhorar o contraste
    img_eq = cv2.equalizeHist(img)

    # Aplicar filtro de mediana para remover ruídos
    img_blur = cv2.medianBlur(img_eq, 5)

    # Aplicar um filtro de nitidez
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_blur, -1, kernel)

    return img, img_sharp  # Retorna a imagem original e a processada

# Caminho da imagem (altere para o caminho correto da sua imagem)
image_path = "raiox.jpeg"

# Pré-processar a imagem
img_original, img_processada = preprocess_image(image_path)

# Exibir imagens antes e depois do pré-processamento
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_original, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_processada, cmap='gray')
plt.title("Pré-processada")
plt.axis("off")

plt.show()
