import os
import cv2
import matplotlib.pyplot as plt

# Caminhos das pastas (Altere conforme necessário)
dataset_path = "archive (4)/dataset"  # Ex: "C:/Users/SeuNome/Dataset"
folders = ["train", "test", "val"]

# Contar imagens por classe
for folder in folders:
    path_fractured = os.path.join(dataset_path, folder, "fractured")
    path_not_fractured = os.path.join(dataset_path, folder, "not fractured")

    count_fractured = len(os.listdir(path_fractured))
    count_not_fractured = len(os.listdir(path_not_fractured))

    print(f"\n📂 {folder.upper()}:")
    print(f"🔴 Fractured: {count_fractured} imagens")
    print(f"🟢 Not Fractured: {count_not_fractured} imagens")

# Mostrar algumas imagens da base
def show_sample_images(folder, class_name, num_images=5):
    path = os.path.join(dataset_path, folder, class_name)
    images = os.listdir(path)[:num_images]

    plt.figure(figsize=(10, 5))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(class_name)
    plt.show()

# Exibir exemplos de imagens
print("\n🔍 Visualizando imagens de treino:")
show_sample_images("train", "fractured")
show_sample_images("train", "not fractured")
