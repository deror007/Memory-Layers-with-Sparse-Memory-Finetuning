import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm
import copy
from src.memory_layer import MemoryPlusLayer


"""
ViT comparison between pre-trained "dense baseline" and "memory augmented" versions,
on CIFAR-100 dataset.
"""

# Use the MemoryPlusLayer class defined in the previous steps

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def run_comparison(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Preparation
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR100(root='./data_dir', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data_dir', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 2. Model Initialization
    # Baseline: Standard Pre-trained ViT
    dense_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100, cache_dir="./models_dir").to(device)
    
    # Memory Augmented: Replace Block 6 (Centered) [cite: 248, 249, 293]
    memory_model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100).to(device)
    d_model = memory_model.embed_dim
    memory_slots = 1024**2 # 1M slots as used in scaling experiments [cite: 244]
    memory_model.blocks[6].mlp = MemoryPlusLayer(d_model=d_model, memory_slots=memory_slots)
    memory_model.to(device)

    # 3. Training Loop
    results = {'dense': {'loss': [], 'acc': []}, 'memory': {'loss': [], 'acc': []}}
    
    for name, model in [('dense', dense_model), ('memory', memory_model)]:
        print(f"Training {name} model...")
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            
            # results[name]['train_loss'].append(train_loss)
            # results[name]['train_acc'].append(train_acc)

            results[name]['val_loss'].append(val_loss)
            results[name]['val_acc'].append(val_acc)
            print(f"Epoch {epoch+1}: Val Acc {val_acc:.2f}%")

    # 4. Plotting
    plt.figure(figsize=(12, 5))
    
    # Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(results['dense']['val_acc'], label='Dense Baseline', marker='o')
    plt.plot(results['memory']['val_acc'], label='Memory+ Adapter', marker='s')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(results['dense']['val_loss'], label='Dense Baseline', marker='o')
    plt.plot(results['memory']['val_loss'], label='Memory+ Adapter', marker='s')
    plt.title('Validation Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()