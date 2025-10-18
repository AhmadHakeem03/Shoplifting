# Shoplifting Detection using Deep Learning

This repository contains two approaches for detecting shoplifting behavior in surveillance videos using deep learning techniques.

## Overview

The project implements binary classification to distinguish between shoplifting and non-shoplifting behavior using video data. Two different architectures are explored:

1. **R(2+1)D CNN** - A pretrained video classification model
2. **Custom 3D CNN** - A lightweight 3D convolutional neural network

## Dataset

- **Shoplifters**: 324 videos
- **Non-shoplifters**: 528 videos
- **Minimum frames**: 75 frames per video
- **Train/Val/Test split**: 80% / 10% / 10%

## Approach 1: R(2+1)D CNN

### Model Architecture
- Pretrained R(2+1)D-18 model from torchvision
- Custom classification head with:
  - Linear(512) + LayerNorm + ReLU + Dropout(0.5)
  - Linear(256) + ReLU + Dropout(0.3)
  - Linear(2) output layer

### Training Details
- **Epochs**: 20
- **Learning rate**: 1e-4
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss
- **Batch size**: 4
- **Data augmentation**: RandomHorizontalFlip, ColorJitter

### Results
- **Final Train Accuracy**: 79.00%
- **Final Validation Accuracy**: 90.59%

## Approach 2: Custom 3D CNN with Data Deduplication

### Key Features
- **ResNet50-based feature extraction** for similarity detection
- **Cosine similarity filtering** to remove near-duplicate videos from training set
- **Lightweight 3D CNN architecture**

### Data Deduplication Process
1. Extract frame embeddings using pretrained ResNet50
2. Calculate cosine similarity between training and validation sets
3. Remove training videos with similarity > 0.996 to validation videos
4. **Result**: Reduced training set from 681 to 122 videos (removed 559 duplicates)

### Model Architecture
- Conv3D(3→64) + MaxPool3D
- Conv3D(64→128) + MaxPool3D
- Conv3D(128→256) + MaxPool3D
- Fully Connected(512) + ReLU
- Output(2 classes)

### Training Details
- **Epochs**: 20
- **Learning rate**: 1e-4
- **Optimizer**: Adam
- **Batch size**: 8
- **Input**: 16 frames sampled uniformly from each video

### Results
- **Final Train Accuracy**: 99.18%
- **Final Validation Accuracy**: 89.41%
- **Test Accuracy**: 93.02%
- **Test Precision**: 89.66%
- **Test Recall**: 89.66%
- **Test F1 Score**: 89.66%

## Requirements
```bash
pip install torch torchvision opencv-python matplotlib numpy pillow tqdm scikit-learn torchmetrics
```

## Usage

### Training R(2+1)D Model
Run the first notebook/script for R(2+1)D implementation.

### Training Custom 3D CNN
Run the second notebook/script for Custom 3D CNN with deduplication. Model saves as "cnn3d_best1.pth".

## Key Insights

1. **Data Quality**: The deduplication step significantly improved model reliability by removing near-duplicate videos that could leak information between train/val sets.

2. **Performance**: Despite using a much smaller training set (122 vs 681 videos), the custom 3D CNN achieved excellent test performance (93% accuracy).

3. **Overfitting**: The high training accuracy (99%+) suggests some overfitting, but validation performance remained strong.

## Future Improvements

- Implement early stopping to prevent overfitting
- Add learning rate scheduling
- Experiment with data augmentation techniques
- Try ensemble methods combining both approaches
- Expand dataset with more diverse scenarios



## Acknowledgments

- PyTorch and torchvision for pretrained models
- Google Colab for GPU resources
