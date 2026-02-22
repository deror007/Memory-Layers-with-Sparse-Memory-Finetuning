"""
ViT Pretrained Ablation Study
----------------------------
This script compares a pretrained ViT Tiny model with and without the MemoryPlusLayer.
"""

import sys
import os

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.getcwd())

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import timm
from src.memory_layer import MemoryPlusLayer

def profile_model_performance(model, device, name="Model"):
    """Profiles a single forward and backward pass to see memory/FLOP tradeoffs."""
    print(f"\n--- Profiling {name} ---")
    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)
    

     # [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
    with profile(
        activities = [ProfilerActivity.CPU],  
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward_pass"):
            output = model(inputs)
        with record_function("backward_pass"):
            loss = output.sum()
            loss.backward()
            
    # Sorted by CUDA time if available, else CPU time
    sort_by = "gpu_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=10))

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        print("hi")
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
    running_loss, correct, total = 0.0, 0, 0
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

def run_comparison(epochs=5):
    # Detect device (Note: MPS for Mac is an option, but profiler support varies)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.FashionMNIST(root='./data_dir', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data_dir', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Init Models
    dense_model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=10, cache_dir="./models_dir").to(device)
    
    memory_model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=10, cache_dir = "./models_dir").to(device)
    d_model = memory_model.embed_dim
    memory_slots = 256**2 
    memory_model.blocks[6].mlp = MemoryPlusLayer(d_model=d_model, memory_slots=memory_slots).to(device)

    # PROFILE MODELS BEFORE TRAINING TO SEE MEMORY/FLOP TRADEOFFS
    # profile_model_performance(dense_model, device, name="Dense Baseline")
    # profile_model_performance(memory_model, device, name="Memory+ Adapter")
    
    """VERY IMPORTANT NOTE: 
    
    Base model is 17x faster than Memory+ (1024**2 memory slots) ViT!!!! 
    NEED CUSTOM KERNEL FOR EMBEDDINGBAG SOLUTION TO SPEED THIS UP, 
    AS THIS IS THE BOTTLENECK IN THE MEMORY LAYER.

    Hoever found memory slot size 256**2 to be near performance of baseline!
    
    """
    # quit()
    
    # Fixed keys to match your storage logic
    results = {'dense': {'val_loss': [], 'val_acc': []}, 'memory': {'val_loss': [], 'val_acc': []}}
    
    for name, model in [('dense', dense_model), ('memory', memory_model)]:
        print(f"\nStarting training for {name}...")
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}: train loss: {train_loss}, train acc: {train_acc}")
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            
            results[name]['val_loss'].append(val_loss)
            results[name]['val_acc'].append(val_acc)
            print(f"Epoch {epoch+1}: Val Acc {val_acc:.2f}%")

    return results


# 5. Plotting Accuracy
def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.plot(results['dense']['acc'], label='Dense Baseline (Pre-trained ViT)')
    plt.plot(results['memory']['acc'], label='Memory+ Adapter ViT')
    plt.title('CIFAR-100 Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    start_time = time.time()
    res = run_comparison()
    end_time = time.time()
    plot_results(res)
    print(f"Run comparison duration: {end_time - start_time:.2f} seconds")