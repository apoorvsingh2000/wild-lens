# WildLens: WildLens: Wildlife Detection & Classification

## Summary:
Comparative Analysis of DenseNet and Vision Transformers for Wildlife Image Classification.

WildLens is a deep learning project aimed at developing an advanced algorithm for detecting and classifying wildlife in their natural habitats using camera trap images. This project explores various deep learning architectures, with a particular focus on DenseNet and Vision Transformers (ViT), to determine the most effective model for this task.

## Project Overview

This project utilizes the iWildCam-2020-FGVC7 dataset to train and evaluate deep learning models capable of classifying wildlife species. The primary goal is to enhance biodiversity monitoring and conservation efforts through improved automated image analysis.

### Key Objectives

- To develop a robust machine learning model for wildlife image classification.
- To compare the performance of DenseNet and Vision Transformers under different configurations.
- To identify the optimal model based on empirical loss metrics over multiple training epochs.

## Getting Started

Follow these steps to set up the project environment and start training the model.

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip or conda
- Kaggle API (for dataset download)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rishavroy97/wild-lens.git
   cd wild-lens
   
2. **Install the required libraries:**
   ```bash
   
   pip install -r requirements.txt

4. **Download the Dataset**
The iWildCam-2020-FGVC7 dataset is approximately 120 GB. Ensure you have enough space and a stable internet connection.

5. **Set up Kaggle API credentials: Follow instructions here to set up your Kaggle API credentials if you haven't already.**
Download and unzip the dataset:
    ```bash
    
    kaggle competitions download -c iwildcam-2020-fgvc7
    unzip iwildcam-2020-fgvc7.zip -d data

4. **Data Preprocessing**
Merge the required data into a single CSV file for training.
    ```bash
    
    python join_data.py

5. **Model Training**
To train the model, use the following command. You can adjust hyperparameters like the number of epochs, batch size, and model name according to your requirements.
    ```bash

    python train_model.py --start 0 --num-epochs 15 --batch-size 64 --save-every 5 --model-name vit_full

### Models Explored
DenseNet: Proved to be the most effective model in this study, particularly due to its efficient use of parameters and ability to reduce overfitting through feature reuse.
Vision Transformers (ViT): Explored with both frozen and unfrozen weights. ViT with frozen weights optimizes only the classifier, while with unfrozen weights, the entire network is trainable.
Performance
DenseNet demonstrated superior performance with consistently lower loss metrics compared to the Vision Transformer models over 15 epochs.

### Conclusion
The findings from this study highlight DenseNet’s architectural advantages, making it suitable for ecological datasets and complex classification tasks. Future work will explore methods to address data imbalances and expand the application of DenseNet to other datasets.
