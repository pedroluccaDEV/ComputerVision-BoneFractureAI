import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# Definindo a arquitetura da CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Supondo que as imagens tenham tamanho 224x224
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flattening
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Probabilidade entre 0 e 1
        return x

# Definir as transformações de pré-processamento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Definindo o dataset e o DataLoader
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset/val", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inicializando o modelo, critério e otimizador
model = CNNModel()  # Usar a CPU, sem transferência para a GPU
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treinamento
def train_model():
    model.train()
    for epoch in range(10):  # 10 épocas
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.float(), labels.float()  # Garantir que os dados sejam do tipo float
            optimizer.zero_grad()
            outputs = model(inputs)

            # Ajustando a forma de outputs e labels para que tenham o mesmo tamanho
            outputs = outputs.view(-1)  # Alterando para formato adequado (1D tensor)
            labels = labels.view(-1)  # Alterando para formato adequado (1D tensor)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Treinar o modelo
train_model()

# Salvar o modelo treinado
torch.save(model.state_dict(), "bone_fracture_detector.pth")
print("✅ Modelo treinado e salvo com sucesso!")
