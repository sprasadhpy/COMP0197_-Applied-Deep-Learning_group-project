from train import train_segmentation_model
import torch
from data_loader_newsplit import get_data_loader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "./"
print(device)

# Get data loaders
train_labeled_loader, train_unlabeled_loader, val_loader, test_loader = get_data_loader()

# Train model with Dice Loss
dice_trained_model = train_segmentation_model(
    train_labeled_loader,
    train_unlabeled_loader,
    test_loader,
    device,
    num_epochs=20,
    lr=1e-3,
    use_dice=True
)
# Save trained model
torch.save(dice_trained_model.state_dict(), f'saved_model_dice.pt')
print(f'Model trained with Dice Loss saved.')

# # Train model with IOU Loss
# iou_trained_model = train_segmentation_model(
#     train_labeled_loader,
#     train_unlabeled_loader,
#     test_loader,
#     device,
#     num_epochs=200,
#     lr=1e-3,
#     use_dice=True
# )
# # Save trained model
# torch.save(iou_trained_model.state_dict(), f'saved_model_iou.pt')
# print(f'Model trained with IOU Loss saved.')