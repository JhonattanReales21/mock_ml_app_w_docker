"""
Training module for CNN models on medical images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Tuple


class MedicalImageDataset(Dataset):
    """
    Custom Dataset class for medical images.
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images (numpy.ndarray): Array of images
            labels (numpy.ndarray): Array of labels
            transform (callable, optional): Optional transform to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).long()
        
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return epoch_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model (nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on (cpu or cuda)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return epoch_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs=10, 
                learning_rate=0.001, device='cpu'):
    """
    Complete training loop for the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        dict: Training history with losses and accuracies
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Training on {device}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    print("-" * 60)
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def save_model(model, filepath):
    """
    Save model to disk.
    
    Args:
        model (nn.Module): Model to save
        filepath (str): Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu'):
    """
    Load model from disk.
    
    Args:
        model (nn.Module): Model instance to load weights into
        filepath (str): Path to the saved model
        device (str): Device to load the model on
        
    Returns:
        nn.Module: Loaded model
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model loaded from {filepath}")
    return model
