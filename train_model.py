import argparse
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from image_datset import ImageDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_DIR = f'./models/vit_full'

# Precomputations
DF = pd.read_csv('./wild-lens.csv')
CLASSES_TO_USE = DF['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict(list(zip(CLASSES_TO_USE, range(NUM_CLASSES))))
IMG_SIZE = 224  # Required size for a Vision Transformer
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZER,
])
TRAIN_IMGS_DIR = './iwildcam-2020-fgvc7/train'
TRAIN_DF, VAL_DF = train_test_split(DF[['file_name', 'category_id']], test_size=0.2, random_state=42, shuffle=True)

# Training Parameters
NUM_EPOCHS = 25
START_EPOCH = 0
BATCH_SIZE = 64
SAVE_EVERY = 5

# Datasets
train_dataset = ImageDataset(TRAIN_DF, TRAIN_IMGS_DIR, n_classes=NUM_CLASSES, label_dict=CLASSMAP,
                             transforms=TRANSFORMATIONS)
val_dataset = ImageDataset(VAL_DF, TRAIN_IMGS_DIR, n_classes=NUM_CLASSES, label_dict=CLASSMAP,
                           transforms=TRANSFORMATIONS)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Loss Calculator
criterion = torch.nn.CrossEntropyLoss()


def calculate_validation_loss(model, val_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    return total_loss / num_samples


def train(model, start=0, num_epochs=10):
    # Model
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
        threshold_mode="rel",
        min_lr=1e-8,
        threshold=0.01,
    )

    # Load checkpoint
    checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{start}.pth'
    if start > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        train_loss = checkpoint['train_loss']
        print(f'Loaded checkpoint at epoch {start}')

    # Load best loss
    best_model_path = f'{BEST_MODEL_DIR}/best_model.pth'
    best_loss = float('inf')
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path)
        best_loss = best_model['loss']

    train_losses = []
    val_losses = []
    epoch_count = 0
    val_loss = float('inf')

    for epoch in range(start, num_epochs):
        # Training
        total_train_loss = 0.0
        num_train_samples = 0
        model.train()
        for images, labels in train_dataloader:
            # Forward Pass
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_train_loss += train_loss.item() * batch_size
            num_train_samples += batch_size

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        lr_scheduler.step(epoch)

        # Validation phase
        val_loss = calculate_validation_loss(model, val_dataloader)

        # Save the losses (every epoch)
        val_losses.append(val_loss)
        epoch_train_loss = total_train_loss / num_train_samples
        train_losses.append(epoch_train_loss)
        epoch_count += 1
        save_losses(train_losses, val_losses, start, epoch_count)

        # Print loss every epoch
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {val_loss:.4f}')

        # Save the best model every 10 epochs
        if epoch % SAVE_EVERY == 0:
            if val_loss < best_loss:
                best_loss = val_loss

                # Save the best model
                if not os.path.exists(BEST_MODEL_DIR):
                    os.makedirs(BEST_MODEL_DIR)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                print(f'Best model saved at epoch {epoch}')

            # Save checkpoint
            if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)

            checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {epoch}")

    print('Training finished!')
    print(f'Final Loss: {val_loss:.4f}')

    return train_losses, val_losses


def save_losses(train_losses, val_losses, start_epoch, num_epochs):
    tl = torch.tensor(train_losses)
    vl = torch.tensor(val_losses)

    checkpoint_path = f'{CHECKPOINT_DIR}/checkpoint_epoch{start_epoch}.pth'
    if start_epoch > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']

    if not os.path.exists(BEST_MODEL_DIR):
        os.makedirs(BEST_MODEL_DIR)

    torch.save({
        'train_losses': tl,
        'validation_losses': vl,
        'start': start_epoch,
        'num_epochs': num_epochs
    }, f"{BEST_MODEL_DIR}/model-losses.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Model for 25 Epoch")
    parser.add_argument("--start", type=int, default=START_EPOCH, help="Starting index epoch")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save every N epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--model-name", type=str, default="vit_full", help="Model directory name")
    args = parser.parse_args()

    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Validation Dataset Size: {len(val_dataset)}")

    BATCH_SIZE = args.batch_size
    BEST_MODEL_DIR = f'./models/{args.model_name}'

    # select model
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads[0] = torch.nn.Linear(model.heads[0].in_features, NUM_CLASSES)

    train_losses, val_losses = train(model, args.start, NUM_EPOCHS)
    save_losses(train_losses, val_losses, args.start, NUM_EPOCHS)
