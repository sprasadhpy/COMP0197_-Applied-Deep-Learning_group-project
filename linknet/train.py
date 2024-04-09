import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
import os
from lossfunction import supervised_dice_loss, supervised_iou_loss, semi_supervised_dice_loss, semi_supervised_iou_loss
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import dice

base_dir = "./"


def train_segmentation_model(train_loader_with_label, train_loader_without_label, test_loader, device, num_epochs=50,
                             lr=1e-4, use_dice=True):
    global base_dir
    """
    Trains a semi-supervised segmentation model using both labeled and unlabeled data.

    Args:
        train_loader_with_label (DataLoader): DataLoader for labeled training data.
        train_loader_without_label (DataLoader): DataLoader for unlabeled training data.
        test_loader (DataLoader): DataLoader for test/validation data.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        use_dice (bool, optional): If True, uses Dice Loss; otherwise, uses IOU Loss. Defaults to True.

    Returns:
        nn.Module: Trained segmentation model.
    """
    sw = SummaryWriter(os.path.join(base_dir, 'logs'))
    step = 0  # For TensorBoard
    # Initialize the neural network
    model = link_net(classes=3).to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)


    experiment = "dice" if use_dice else "iou"
    if use_dice:
        print("Training with Dice Loss function...")
    else:
        print("Training with IOU Loss function...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        examples = 0
        # Train on both labeled and unlabeled data during each epoch of training
        train_iter_without_label = iter(train_loader_without_label)
        for i, (images_with_label, labels) in enumerate(train_loader_with_label):
            try:
                images_without_label, _ = next(train_iter_without_label)
            except StopIteration:
                train_iter_without_label = iter(train_loader_without_label)
                images_without_label, _ = next(train_iter_without_label)

            images_with_label, labels = images_with_label.to(device), labels.to(device)
            images_without_label = images_without_label.to(device)

            # Set alpha based on i (or should it be epoch number?)
            t1 = 100
            t2 = 600
            if i < t1:
                alpha = 0
            elif i < t2:
                alpha = (i - t1) / (t2 - t1)
            else:
                alpha = 3

            # Zero the parameter gradients
            optimizer.zero_grad()

            # print('images_with_label', images_with_label.size())
            # print('labels', labels.size())

            # Forward pass, backward pass, and optimization
            pred_with_label = model(images_with_label)
            pred_without_label = model(images_without_label)

            loss = criterion(pred_with_label, labels)

            # # Determine the loss function used
            # if use_dice:
            #     loss = semi_supervised_dice_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            # else:
            #     loss = semi_supervised_iou_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            examples = examples + images_with_label.size(0)
            running_loss += loss.item() * images_with_label.size(0)
            loss.backward()
            optimizer.step()

            # Print stats every iteration
            # if i %100 == 0:
            #     print(f"Epoch {epoch + 1}, iteration {i + 1}: Loss = {loss.item():.6f}, Alpha = {alpha}")
            step += 1

        training_loss = running_loss / examples
        print(f'Epoch {epoch + 1}, Training Loss: {training_loss:.6f}')
        # Add training loss to TensorBoard
        sw.add_scalar(f"training/{experiment}", running_loss / examples, step)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(base_dir, f"models/{experiment}_model_{epoch}.pth"))

        # Evaluate the model on the test set
        model.eval()

        test_loss = 0
        total_score = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).detach()
                loss = criterion(output, target)
                # if use_dice:
                #     loss = supervised_dice_loss(output, target)
                # else:
                #     loss = supervised_iou_loss(output, target)
                score = 1 - loss.item()
                test_loss += loss.item()
                total_score += score

            _, pred = torch.max(output, 1)
            dice_score = dice(pred, target, average='macro', num_classes=3)


        sw.add_scalar(f"testing/{experiment}_loss", test_loss, step)
        sw.add_scalar(f"testing/{experiment}_score", total_score, step)

        print(
            f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.6f}, Dice Score: {dice_score:.6f}')
        # Delete unnecessary variables
        del data, target, output

        # Empty cache
        torch.cuda.empty_cache()
        # if use_dice:
        #     print(
        #         f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.6f}, Dice Score: {total_score / len(test_loader):.6f}')
        # else:
        #     print(
        #         f'Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader):.6f}, IoU Score: {total_score / len(test_loader):.6f}')

    print("Training completed.")

    return model