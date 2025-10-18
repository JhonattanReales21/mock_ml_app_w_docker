"""
Inference module for making predictions with trained CNN models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def predict_single_image(model, image, device='cpu'):
    """
    Make a prediction for a single image.
    
    Args:
        model (nn.Module): Trained model
        image (numpy.ndarray or torch.Tensor): Input image
        device (str): Device to run inference on
        
    Returns:
        tuple: (predicted_class, confidence_scores)
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor if numpy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), probabilities.cpu().numpy()[0]


def predict_batch(model, images, device='cpu', batch_size=32):
    """
    Make predictions for a batch of images.
    
    Args:
        model (nn.Module): Trained model
        images (numpy.ndarray or torch.Tensor): Batch of images
        device (str): Device to run inference on
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (predictions, confidence_scores)
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor if numpy array
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    
    all_predictions = []
    all_confidences = []
    
    # Process in batches
    num_images = images.shape[0]
    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_confidences.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_confidences)


def predict_with_uncertainty(model, image, device='cpu', num_samples=10):
    """
    Make predictions with uncertainty estimation using Monte Carlo dropout.
    
    Args:
        model (nn.Module): Trained model with dropout layers
        image (numpy.ndarray or torch.Tensor): Input image
        device (str): Device to run inference on
        num_samples (int): Number of forward passes for uncertainty estimation
        
    Returns:
        dict: Dictionary with prediction, confidence, and uncertainty
    """
    model.train()  # Keep dropout active
    model = model.to(device)
    
    # Convert to tensor if numpy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())
    
    predictions = np.array(predictions).squeeze()
    
    # Calculate mean prediction and uncertainty (standard deviation)
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    predicted_class = np.argmax(mean_prediction)
    confidence = mean_prediction[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'mean_probabilities': mean_prediction,
        'uncertainty': uncertainty,
        'all_predictions': predictions
    }


def get_top_k_predictions(probabilities, class_names=None, k=3):
    """
    Get top K predictions with their probabilities.
    
    Args:
        probabilities (numpy.ndarray): Array of class probabilities
        class_names (list, optional): List of class names
        k (int): Number of top predictions to return
        
    Returns:
        list: List of tuples (class_index/name, probability)
    """
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    
    results = []
    for idx in top_k_indices:
        class_label = class_names[idx] if class_names else idx
        results.append((class_label, probabilities[idx]))
    
    return results


def explain_prediction(model, image, predicted_class, device='cpu'):
    """
    Generate a simple explanation for the prediction using gradient-based attribution.
    
    Args:
        model (nn.Module): Trained model
        image (torch.Tensor): Input image
        predicted_class (int): Predicted class index
        device (str): Device to run on
        
    Returns:
        numpy.ndarray: Attribution map
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor if numpy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    image.requires_grad = True
    
    # Forward pass
    outputs = model(image)
    
    # Backward pass for the predicted class
    model.zero_grad()
    outputs[0, predicted_class].backward()
    
    # Get gradients
    gradients = image.grad.data.cpu().numpy()
    
    # Simple attribution: absolute value of gradients
    attribution = np.abs(gradients).squeeze()
    
    return attribution


def inference_pipeline(model, image_path, preprocessing_func, 
                      class_names=None, device='cpu'):
    """
    Complete inference pipeline from image path to prediction.
    
    Args:
        model (nn.Module): Trained model
        image_path (str): Path to the image
        preprocessing_func (callable): Function to preprocess the image
        class_names (list, optional): List of class names
        device (str): Device to run inference on
        
    Returns:
        dict: Dictionary with prediction results
    """
    # Preprocess image
    image = preprocessing_func(image_path)
    
    # Make prediction
    predicted_class, probabilities = predict_single_image(model, image, device)
    
    # Get top K predictions
    top_predictions = get_top_k_predictions(probabilities, class_names, k=3)
    
    # Prepare result
    result = {
        'predicted_class': class_names[predicted_class] if class_names else predicted_class,
        'confidence': probabilities[predicted_class],
        'all_probabilities': probabilities,
        'top_predictions': top_predictions
    }
    
    return result
