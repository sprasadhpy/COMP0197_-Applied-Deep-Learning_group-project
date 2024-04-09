import torchvision

from train import train_segmentation_model, evaluate_model
import torch
from data_loader_newsplit import get_data_loader
from linknet import link_net
import torch.nn.functional as F

import os

def generate_and_save_predictions(test_loader, device, model_path, output_directory):
    # Get a batch of test data
    images, labels = next(iter(test_loader))

    # Move images to the device
    images = images.to(device)
    labels = labels.to(device)

    # Load the state dictionary from the best model
    state_dict = torch.load(model_path)

    trained_model = link_net(classes=3).to(device)
    # Load the state dictionary into the trained_model
    trained_model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    trained_model.eval()
    # Generate predictions
    predictions = trained_model(images)

    _, predicted_masks = torch.max(predictions, 1)

    # Save the results in Output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the images, ground truth masks, and predicted masks
    for i in range(images.shape[0]):
        torchvision.utils.save_image(images[i], os.path.join(output_directory, f"image_{i}.png"))

        # Round predictions to nearest integer and ensure they are single channel images and of type Float
        predictions_single_channel = torch.round((predicted_masks[i].unsqueeze(0).float() / 3.0) )  # normalize to [0,255] for save_image

        # Ensure labels are single channel images and of type Float
        labels_single_channel = torch.round((labels[i].unsqueeze(0).float() / 3.0))  # normalize to [0,255] for save_image

        torchvision.utils.save_image(labels_single_channel, os.path.join(output_directory, f"label_{i}.png"))
        torchvision.utils.save_image(predictions_single_channel, os.path.join(output_directory, f"prediction_{i}.png"))




def main():

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    print(device)

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loader(BATCH_SIZE)

    # Train model with Dice Loss
    trained_model = train_segmentation_model(
        train_loader,
        val_loader,
        device,
        num_epochs=100,
        lr=1e-5,
    )

    #save the best model to models directory
    torch.save(trained_model.state_dict(), os.path.join("models", "best_model.pth"))

    # Evaluate model on test set

    dice_score, iou_score = evaluate_model(trained_model, test_loader, device)


    generate_and_save_predictions(test_loader, device, os.path.join("models", "best_model.pth"), "Output")

if __name__ == "__main__":
    main()

