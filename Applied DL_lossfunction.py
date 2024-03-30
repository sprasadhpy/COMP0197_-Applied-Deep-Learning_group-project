import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex as JI

def calculate_supervised_dice_loss(predictions, ground_truth):
    """
    Calculate the supervised Dice loss for labeled data.

    Args:
        predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised Dice loss.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    predictions = torch.sigmoid(predictions)
    
    # Smoothing factor to prevent division by zero
    smooth = 1e-5
    
    # Compute the intersection and union
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    union = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3))
    
    # Calculate the Dice loss
    dice_loss = 1 - 2 * (intersection + smooth) / (union + smooth)
    
    # Average the losses across the batch
    return dice_loss.mean()

def calculate_supervised_iou_loss(predictions, ground_truth):
    """
    Calculate the supervised IOU loss for labeled data.

    Args:
        predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised IOU loss.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    predictions = torch.sigmoid(predictions)
    
    # Smoothing factor to prevent division by zero
    smooth = 1e-5
    
    # Compute the intersection and union
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    union = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3))
    
    # Calculate the IOU loss
    iou_loss = 1 - (intersection + smooth) / (union + smooth)
    
    # Calculate IOU using torchmetrics.JaccardIndex
    iou = JI(task="multiclass", num_classes=38)
    iou_loss_library = iou(predictions, ground_truth)
    print("The IoU loss is : {}", 1 - iou_loss_library, iou_loss.mean())
    
    # Average the losses across the batch
    return iou_loss.mean()

def generate_pseudo_labels(unlabeled_predictions):
    """
    Generate pseudo-labels from the unlabeled predictions.
    
    Args:
        unlabeled_predictions (torch.Tensor): Predictions with shape (batch_size, num_classes, height, width).
        
    Returns:
        torch.Tensor: Pseudo-labels with shape (batch_size, num_classes, height, width), where the channel with the 
        highest likelihood for each pixel is set to 1, and all other channels are set to 0.
    """
    # Get the channel with the highest likelihood for each pixel
    max_channels = torch.argmax(unlabeled_predictions, dim=1, keepdim=True)
    
    # Create a tensor filled with zeros, matching the shape of unlabeled_predictions
    pseudo_labels = torch.zeros_like(unlabeled_predictions)
    
    # Set the channel with the highest likelihood for each pixel to 1
    pseudo_labels.scatter_(1, max_channels, 1)
    
    return pseudo_labels

def calculate_semi_supervised_dice_loss(labeled_predictions, ground_truth, unlabeled_predictions, alpha=0.5):
    """
    Calculate the semi-supervised Dice loss for labeled and unlabeled data using pseudo-labels.

    Args:
        labeled_predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).
        unlabeled_predictions (torch.Tensor): Predictions for unlabeled data, with shape (batch_size, num_classes, height, width).
        alpha (float): Weight for consistency regularization. Default is 0.5.

    Returns:
        torch.Tensor: Semi-supervised Dice loss.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    labeled_predictions = torch.sigmoid(labeled_predictions)
    
    # Compute labeled loss
    labeled_loss = calculate_supervised_dice_loss(labeled_predictions, ground_truth)
    
    # Generate pseudo-labels from unlabeled data
    unlabeled_predictions_pseudo = generate_pseudo_labels(unlabeled_predictions)
    
    # Compute unlabeled loss
    unlabeled_loss = calculate_supervised_dice_loss(unlabeled_predictions, unlabeled_predictions_pseudo)

    # Combine the losses
    return labeled_loss + alpha * unlabeled_loss

def calculate_semi_supervised_iou_loss(labeled_predictions, ground_truth, unlabeled_predictions, alpha=0.5):
    """
    Calculate the semi-supervised IOU loss for labeled and unlabeled data using pseudo-labels.

    Args:
        labeled_predictions (torch.Tensor): Predictions for labeled data, with batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).
        unlabeled_predictions (torch.Tensor): Predictions for unlabeled data, with shape (batch_size, num_classes, height, width).
        alpha (float): Weight for consistency regularization. Default is 0.5.

    Returns:
        torch.Tensor: Semi-supervised IOU loss.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    labeled_predictions = torch.sigmoid(labeled_predictions)
    
    # Compute labeled loss
    labeled_loss = calculate_supervised_iou_loss(labeled_predictions, ground_truth)
    
    # Generate pseudo-labels from unlabeled data
    unlabeled_predictions_pseudo = generate_pseudo_labels(unlabeled_predictions)
    
    # Compute unlabeled loss
    unlabeled_loss = calculate_supervised_iou_loss(unlabeled_predictions, unlabeled_predictions_pseudo)

    # Combine the losses
    return labeled_loss + alpha * unlabeled_loss
