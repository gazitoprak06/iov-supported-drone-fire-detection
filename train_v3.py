import argparse
import os
import random
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from pathlib import Path
import platform

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed):
    """Seed every generator the training path draws from.

    Without this the run is not reproducible at all: the shuffling of the
    training loader, the brightness and contrast jitter, and the initialisation
    of the two-class head all draw from unseeded generators, so two runs of this
    script produce different weights. The manuscript describes the reported run
    as single-seed, which is only meaningful if a seed is actually fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU convolutions are already deterministic


def main(seed=0, out_name="v3_mobilenet.pth", quiet=False):
    set_seed(seed)
    # Setup paths
    base_dir = "Proje_Kodlari/data/dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    out_dir = 'Proje_Kodlari/evaluation_results/v3_deep_edge'
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cpu")
    if not quiet:
        print(f"Using device: {device}  seed: {seed}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    
    batch_size = 32
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace classifier
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    epochs = 3
    history = []
    
    best_model_path = os.path.join(out_dir, out_name)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double().item() / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double().item() / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", flush=True)
        torch.save(model.state_dict(), best_model_path)
        
    print("Done training!")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the loader shuffle, the augmentation and the head init")
    ap.add_argument("--out", default="v3_mobilenet.pth",
                    help="weight filename written under the deep edge results directory")
    a = ap.parse_args()
    main(seed=a.seed, out_name=a.out)
