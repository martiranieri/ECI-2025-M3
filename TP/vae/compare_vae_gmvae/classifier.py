import os
from PIL import Image
import torch
from torchvision import transforms
import sys
import os

import torch.nn as nn

# Agregar ruta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codebase.models.nns.v1 import Classifier

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hiperparámetros
batch_size = 128
epochs = 5
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Datasets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_set = datasets.MNIST(root='../data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Modelo
classifier = Classifier(y_dim=10).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        optimizer.zero_grad()
        logits = classifier(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluación rápida
classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        preds = classifier(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Test accuracy: {correct / total:.4f}")

# Guardar checkpoint
torch.save({'model_state_dict': classifier.state_dict()}, 'classifier_mnist.pt')
