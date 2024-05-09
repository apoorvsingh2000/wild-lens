import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from image_datset import ImageDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model details
MODEL_NAME = 'vit_full'
BEST_MODEL_DIR = f'./models/{MODEL_NAME}'

# Precomputations
DF = pd.read_csv('./wild-lens.csv')
CLASSES_TO_USE = DF['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict(list(zip(CLASSES_TO_USE, range(NUM_CLASSES))))
REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])

# Image pre-processing
IMG_SIZE = 224
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZER,
])

# Image Data
TRAIN_IMGS_DIR = './iwildcam-2020-fgvc7/train'
TRAIN_DF, VAL_DF = train_test_split(DF[['file_name', 'category_id']], test_size=0.2, random_state=42, shuffle=True)

# Validation Parameters
NUM_WORKERS = 8
BATCH_SIZE = 64

# VAL DATASET
val_dataset = ImageDataset(VAL_DF, TRAIN_IMGS_DIR, n_classes=NUM_CLASSES, label_dict=CLASSMAP,
                           transforms=TRANSFORMATIONS)


def get_category_df():
    with open('./iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as f:
        data = json.load(f)
    return pd.DataFrame(data['categories'])


def calculate_validation_predictions(model, val_loader):
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in val_loader:
            # Forward Pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            outputs = (torch.max(torch.exp(outputs), 1)[1]).cpu().numpy()
            y_pred.extend(outputs)

            labels = (torch.max(labels, 1)[1]).cpu().numpy()
            y_true.extend(labels)

    return y_pred, y_true


def get_cat(id, categories):
    class_id = REVERSE_CLASSMAP[id]
    return categories[categories['id'] == class_id]['name'].values[0]


def generate_confusion_matrix(model):
    # Data Loader
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True)

    model.to(device)

    best_model_path = f'{BEST_MODEL_DIR}/best_model.pth'
    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model['model_state_dict'])

    y_pred, y_true = calculate_validation_predictions(model, val_dataloader)
    class_df = get_category_df()

    unique_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    headers = [get_cat(elem, class_df) for elem in unique_classes]

    # Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)

    df_cm = pd.DataFrame(cf_matrix, index=headers, columns=headers)

    df_cm.to_csv(f'{MODEL_NAME}_confusion_matrix.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrix for the model")

    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Number of threads used by dataloader")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--model-name", type=str, default="vit_full", help="Model directory name")
    args = parser.parse_args()

    print(f"Validation Dataset Size: {len(val_dataset)}")

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    MODEL_NAME = args.model_name
    BEST_MODEL_DIR = f'./models/{MODEL_NAME}'

    # select model
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    model.heads[0] = torch.nn.Linear(model.heads[0].in_features, NUM_CLASSES)

    generate_confusion_matrix(model)
