import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset import DogBreedDataset
from model import get_model

# Paths
DATA_DIR = './data/dog-breed-identification/train'
LABELS_CSV = './data/dog-breed-identification/labels.csv'
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = '../api/resnet18_dogbreed.pt'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = DogBreedDataset(DATA_DIR, LABELS_CSV, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=len(train_dataset.breeds)).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save
torch.save(model.state_dict(), MODEL_PATH)