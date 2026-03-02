"""
------- Sparse Memory Finetuning --------
Take a memory layer adapted model, and finetune it using the 
SMF approach via Lin et al. (2025).
"""

from src.memory_layer import MemoryPlusLayer
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import timm
import math
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader, Subset

# PYTORCH_ALLOC_CONF=expandable_segments:True enabled to prevent fragmentation during this secondary training phase. For Google Colab.

"""
TODO:
1. [] Implement Sparse Memory finetuning using Lin et. al (2025) paper.
    - [] Modify this so that only the forward activations from Memory Layer onwards gets stored!
    - [] Progressive Data-dropout implementation.
2. [x] create training set from fashion mnist dataset by only including misclassifications.
3. [] Get Google Colab!
4. [] compare performance to pre-trained baseline memory model.
5. [] probably use avalanche for split long-tail classification dataset, and redo finetuning on different classes.

"""


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    # Init Device
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def create_tiny_vit_memory_model(device):
    # assuming fashion_mnist is what it was trained on
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10, cache_dir = "./models_dir").to(device)
    d_model = model.embed_dim
    memory_slots = 256**2
    model.blocks[6].mlp = MemoryPlusLayer(d_model=d_model, memory_slots=memory_slots).to(device)
    return model


def load_tiny_vit_memory_model(file_path, device):
    model = create_tiny_vit_memory_model(device)
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    return model


def create_finetune_datasets(model, loader, device):
    model.eval()
    misclassified_indices = []
    correct_indices = []
    
    print("Identifying hard examples in validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Determine which samples in the batch were wrong
            mask = predicted.ne(labels)
            
            # Convert batch-relative indices to global dataset indices
            batch_start = i * loader.batch_size
            for j, is_wrong in enumerate(mask):
                global_idx = batch_start + j
                if is_wrong:
                    misclassified_indices.append(global_idx)
                else:
                    correct_indices.append(global_idx)
                    
    # Create the Subsets
    # Use the original dataset from the loader
    dataset = loader.dataset
    finetune_set = Subset(dataset, misclassified_indices)
    stable_set = Subset(dataset, correct_indices)
    
    print(f"Extraction complete:\n   {len(finetune_set)} misclassified samples found.\n   {len(stable_set)} classified samples found. ")
    
    return finetune_set, stable_set

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # set tqdm on loader for a visual progress bar
    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)

    for images, labels in pbar:

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

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })

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



def sparse_memory_finetuning(model, train_loader, test_loader, device, epochs, seed):
    # Detect device (Note: MPS for Mac is an option, but profiler support varies)

    # print(f"\nStarting training for {name}...")

    trial_accs = []
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: train loss: {train_loss}, train acc: {train_acc}")
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Val Acc {val_acc:.2f}%")
        trial_accs.append(val_acc)


    end_time = time.time()
    speed = end_time - start_time 


    # # del model, optimizer <- do this memory cleanup after and before when doing it multiple times.s
    # gc.collect()
    # torch.cuda.empty_cache()



    return model,train_acc, val_acc, speed