import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import dice
from torch.optim.lr_scheduler import ReduceLROnPlateau

base_dir = os.path.dirname(os.path.abspath(__file__))


def iou(output, target, num_classes):
    smooth = 1e-6
    ious = []

    for cls in range(num_classes):
        output_cls = output == cls
        target_cls = target == cls

        intersection = (output_cls & target_cls).sum()
        union = (output_cls | target_cls).sum()

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return sum(ious) / num_classes




def train_segmentation_model(train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    sw = SummaryWriter(os.path.join(base_dir, 'logs'))
    step = 0  # For TensorBoard

    # Initialize the neural network
    model = link_net(classes=3).to(device)

    # Define loss function and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    min_val_loss = float('inf')
    best_model = None

    # Counter for early stopping
    no_improve_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        examples = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            prediction = model(images)

            loss = criterion(prediction, labels)

            examples = examples + images.size(0)
            running_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()

            step += 1

        training_loss = running_loss / examples
        print(f'Epoch {epoch + 1}, Training Loss: {training_loss:.6f}')
        sw.add_scalar(f"training/loss", training_loss, epoch + 1)
        sw.add_scalar(f"training/lr", optimizer.param_groups[0]['lr'], epoch + 1)

        # Evaluate the model on the test set
        model.eval()

        # save the model every 20 epochs in models directory
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(base_dir, f"models/linknet_{epoch}.pth"))

        with torch.no_grad():
            vali_loss = 0
            total_dice_score = 0
            total_iou_score = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).detach()
                loss = criterion(output, target)
                vali_loss += loss.item()

                _, pred = torch.max(output, 1)
                dice_score = dice(pred, target, average='macro', num_classes=3)
                total_dice_score += dice_score

                iou_score = iou(pred, target, num_classes=3)
                total_iou_score += iou_score

        average_dice_score = total_dice_score / len(val_loader)
        average_iou_score = total_iou_score / len(val_loader)

        validation_loss = vali_loss / len(val_loader)
        sw.add_scalar(f"validation/loss", validation_loss, epoch)
        sw.add_scalar(f"validation/dice_score", average_dice_score, epoch)
        sw.add_scalar(f"validation/iou_score", average_iou_score, epoch)

        print(
            f'Epoch {epoch + 1}, validation_loss: {validation_loss:.6f}, Dice Score: {average_dice_score:.6f}, IoU Score: {average_iou_score:.6f}')

        # If the current model has the lowest validation loss, save it
        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            best_model = model.state_dict()
            no_improve_counter = 0  # Reset counter
        else:
            no_improve_counter += 1  # Increment counter

        # If validation loss didn't improve for more than 10 epochs, stop training
        if no_improve_counter > 10:
            print("Early stopping triggered.")
            break

        # Adjust learning rate
        scheduler.step(validation_loss)

        # Delete unnecessary variables
        del data, target, output

        # Empty cache
        torch.cuda.empty_cache()

    print("Training completed.")

    # Return the model with the lowest validation loss
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    total_dice_score = 0
    total_iou_score = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data).detach()

        _, pred = torch.max(output, 1)
        dice_score = dice(pred, target, average='macro', num_classes=3)
        total_dice_score += dice_score

        iou_score = iou(pred, target, num_classes=3)
        total_iou_score += iou_score

    average_dice_score = total_dice_score / len(test_loader)
    average_iou_score = total_iou_score / len(test_loader)

    print(f'Test Dice Score: {average_dice_score:.6f}, Test IoU Score: {average_iou_score:.6f}')
    return average_dice_score, average_iou_score