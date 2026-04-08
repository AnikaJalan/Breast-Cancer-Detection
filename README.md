# Breast Cancer Detection using Deep Learning

## Overview
This project focuses on detecting breast cancer (benign vs malignant) using deep learning models on histopathological images. It leverages multiple pre-trained CNN architectures and compares their performance under different data augmentation strategies.

The goal is to build an efficient and scalable AI-based diagnostic support system.

---

## Dataset
- Dataset Used: BreaKHis (Breast Cancer Histopathological Dataset)
- Source: Kaggle
- Download Method: kagglehub
- Classes:
  - Benign (0)
  - Malignant (1)

---

## Tech Stack
- Python
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- tqdm

---

## Features
- Automated dataset download using KaggleHub
- Custom PyTorch Dataset class for efficient data handling
- Stratified train-test split
- Implementation of multiple CNN architectures:
  - ResNet50
  - DenseNet121
  - EfficientNet-B0
  - MobileNetV2
  - VGG16
- Transfer Learning (freezing base layers)
- Multiple data augmentation strategies:
  - Basic preprocessing
  - Horizontal flipping
  - Rotation
  - Strong augmentation
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
```

## Usage
1. Download Dataset
```bashimport kagglehub
path = kagglehub.dataset_download("ambarish/breakhis")
print("Dataset downloaded at:", path)
```
2. Run the Notebook

Open and run:
```bash
Breast_Cancer_Detection.ipynb
```
## Workflow
- Data Preprocessing
  - Load image paths from dataset
  - Assign labels (0: benign, 1: malignant)
  - Perform stratified train-test split
- Dataset Class
  - Custom BreakHisDataset class using PyTorch
  - Applies transformations dynamically
- Data Augmentation
  - Resize images to 224x224
  - Apply flipping, rotation, and stronger augmentations
- Model Training
  - Load pretrained models from torchvision
  - Replace final classification layers
  - Freeze base layers for transfer learning
- Training Setup
  - Loss Function: CrossEntropyLoss
  - Optimizer: Adam
  - Batch processing using DataLoader
- Evaluation
  - Predict on test dataset
  - Compute:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix


## Output
- Model performance comparison across architectures
- Impact of different augmentation techniques
- CSV file (breakhis_results.csv) containing:
- Model name
  - Augmentation type
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - TP, TN, FP, FN

## Key Insights
- Transfer learning improves model performance significantly
- Data augmentation enhances generalization capability
- Different architectures behave differently under varying augmentations
- Future Improvements
- Hyperparameter tuning
- Fine-tuning deeper layers instead of freezing all layers
- Experimenting with custom CNN architectures
- Deployment using Flask or FastAPI
- Integration into a real-world clinical decision support system
