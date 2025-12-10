import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from torchvision import transforms

class MNISTTrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filename, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_filename, skiprows=0)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        image = np.array(list(self.data.iloc[idx, 1:]), dtype=np.float32)
        image = image / 255.0  # Normalizar por 255 ao invés de max
        image = image.reshape((28, 28))
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).unsqueeze(0)  # Adicionar canal
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
    def __len__(self):
        return len(self.data)

class MNISTTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filename, transform=None):
        self.data = pd.read_csv(csv_filename, skiprows=0)
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(list(self.data.iloc[idx, :]), dtype=np.float32)
        image = image / 255.0
        image = image.reshape((28, 28))
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).unsqueeze(0)
        return image
        
    def __len__(self):
        return len(self.data)

# Data Augmentation 
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),  # rotation
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),  # translation
        scale=(0.9, 1.1),  #  add zoom
        shear=10  # add shear
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # Random erasing
])

# No augmentation for validation and test
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

#Load datasets with transforms
full_dataset = MNISTTrainDataset('train.csv', transform=train_transform)

# Split training/validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#Apply transform diferently for validation
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Modelo CNN with Batch Normalization
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Adicionar dimensão de canal se necessário
        
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instanciar modelo, loss e otimizador
model = MNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Treinamento
epochs = 10
best_val_accuracy = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validação
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    # Atualizar learning rate
    scheduler.step(accuracy)
    
    # Salvar melhor modelo
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val Accuracy: {accuracy:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')

print(f'\nMelhor accuracy de validação: {best_val_accuracy:.2f}%')

# Carregar melhor modelo para predição
model.load_state_dict(torch.load('best_model.pt'))

# Teste
test_dataset = MNISTTestDataset('test.csv', transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64)

model.eval()
predictions = []
with torch.no_grad():
    for batch_X in test_loader:
        outputs = model(batch_X)
        pred = outputs.argmax(dim=1)
        predictions.extend(pred.numpy())

# Arquivo de submissão
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1), 
    'Label': predictions
})
submission.to_csv('submission.csv', index=False)
print(f'\nSubmission file created with {len(predictions)} predictions')