# LUNG-GENESIS: Enhancing Lung Cancer Histopathological Image Classification Using DCGAN and KATANA

## Overview

**LUNG-GENESIS** is a project aimed at improving the classification of lung cancer histopathological images by integrating Deep Convolutional Generative Adversarial Networks (**DCGAN**) with a Knowledge-Aided Topology and Neural Architecture (**KATANA**). This approach leverages synthetic data generation and domain knowledge to enhance the performance and generalization capabilities of the classification model.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [Baseline Model](#baseline-model)
  - [DCGAN](#dcgan)
    - [Generator Network](#generator-network)
    - [Discriminator Network](#discriminator-network)
  - [KATANA Model](#katana-model)
- [Training Procedures](#training-procedures)
  - [Training DCGAN](#training-dcgan)
  - [Training Baseline Model](#training-baseline-model)
  - [Training KATANA Model](#training-katana-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Quantitative Results](#quantitative-results)
  - [Qualitative Results](#qualitative-results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Dependencies](#dependencies)
- [References](#references)

---

## Introduction

Lung cancer is the leading cause of cancer-related deaths worldwide. Early and accurate diagnosis through histopathological image analysis is crucial for effective treatment planning and improving patient survival rates. However, challenges such as limited annotated datasets and overlapping features among cancer subtypes make this task difficult.

This project introduces **LUNG-GENESIS**, a hybrid approach that:

- **Augments the training data** using **DCGAN** to generate synthetic histopathological images, enriching data diversity.
- **Integrates domain knowledge** through the **KATANA** model to enhance interpretability and performance.
- **Maintains test set purity** by partitioning the test set before any data augmentation to prevent data leakage.

---

## Project Structure

- `LUNG-GENESIS.ipynb`: Main Jupyter notebook containing the implementation.
- `data/`: Directory containing the dataset.
- `models/`: Directory to save and load trained models.
- `results/`: Directory for output results, figures, and evaluation metrics.
- `README.md`: Project documentation and explanations.
- `requirements.txt`: List of required Python packages.

---

## Dataset Description

The dataset used is the **Lung Cancer Histopathological Images** dataset, which includes 15,000 images across three classes:

- **Adenocarcinoma**
- **Benign Tissue**
- **Squamous Cell Carcinoma**

The dataset is split into:

- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

The test set is partitioned before any data augmentation to ensure the integrity of the evaluation process.

---

## Data Preprocessing

- **Data Cleaning**: Removal of duplicates and corrupted images.
- **Image Resizing**: All images resized to 64 × 64 pixels.
- **Normalization**: Applied channel-wise normalization to zero mean and unit variance.
- **Data Augmentation**: Used DCGAN to generate synthetic images, augmenting the training set.
- **Knowledge Feature Extraction**: Calculated mean intensity (μ), variance (σ²), and contrast (C) from grayscale images to capture texture information.

---

## Model Architectures

### Baseline Model

A simple Artificial Neural Network (ANN) used for comparative analysis.

- **Input Layer**: Flattened image pixels.
- **Hidden Layers**: Two fully connected layers with ReLU activation.
- **Output Layer**: Softmax activation for classification into three classes.

### DCGAN

DCGAN consists of two components:

#### Generator Network

- **Input**: Random noise vector (latent space) and one-hot encoded class label.
- **Architecture**:
  - Fully connected layer reshaped to a feature map.
  - Series of transposed convolutional layers with batch normalization and ReLU activation.
  - Output layer with Tanh activation to generate synthetic images.

#### Discriminator Network

- **Input**: Image and class label.
- **Architecture**:
  - Series of convolutional layers with LeakyReLU activation.
  - Flattened output passed through a sigmoid function to classify images as real or fake.

### KATANA Model

Combines Convolutional Neural Networks (CNNs) with a Knowledge-Aided Network (KAN) for incorporating domain-specific knowledge.

- **CNN Pathway**:
  - Extracts spatial features from input images using convolutional layers, ReLU activation, and max pooling.
- **KAN Pathway**:
  - Processes knowledge-based features (μ, σ², C) through fully connected layers.
- **Feature Fusion**:
  - Concatenates outputs from CNN and KAN pathways.
  - Passes through fully connected layers for final classification.

---

## Training Procedures

### Training DCGAN

- **Dataset**: Trained solely on the training set.
- **Objective**: Generator aims to produce realistic images; discriminator aims to distinguish real from fake images.
- **Loss Functions**:
  - **Generator Loss**: Encourages generator to produce images that discriminator classifies as real.
  - **Discriminator Loss**: Measures the ability to distinguish real images from fake ones.
- **Hyperparameters**:
  - Learning Rate: 0.0002
  - Batch Size: 64
  - Epochs: 200
  - Optimizer: Adam

### Training Baseline Model

- **Dataset**: Original training set without synthetic data.
- **Architecture**: Simple ANN as described above.
- **Loss Function**: Categorical Cross-Entropy
- **Hyperparameters**:
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 50
  - Optimizer: Adam

### Training KATANA Model

- **Dataset**: Augmented training set with synthetic images from DCGAN.
- **Loss Function**: Categorical Cross-Entropy
- **Early Stopping**: Based on validation loss to prevent overfitting.
- **Hyperparameters**:
  - Learning Rate: 0.0002
  - Batch Size: 32
  - Epochs: 50
  - Optimizer: Adam

---

## Evaluation Metrics

- **Accuracy**: Percentage of correct predictions over total predictions.
- **Macro F1 Score**: Harmonic mean of precision and recall, calculated per class and averaged.
- **Confusion Matrix**: Visual representation of the performance of the classification model.

---

## Results

### Quantitative Results

| Model                                   | Validation Accuracy | Test Accuracy | Macro F1 Score |
|-----------------------------------------|---------------------|---------------|----------------|
| Baseline Model                          | 86.80%              | 85.50%        | 0.8677         |
| Baseline Model + DCGAN                  | 87.50%              | 86.20%        | 0.8721         |
| KATANA Model                            | 93.56%              | 92.10%        | 0.9356         |
| **LUNG-GENESIS (KATANA + DCGAN)**       | **95.47%**          | **94.15%**    | **0.9546**     |

### Qualitative Results

- **Sample Predictions**: Visualization of model predictions on test images, highlighting correct and incorrect classifications.
- **Synthetic Images**: Comparison between real images and synthetic images generated by DCGAN to demonstrate the quality of generated data.

---

## Conclusion

- **Performance Improvement**: Integrating DCGAN-generated synthetic data and domain knowledge via KATANA significantly improved classification performance.
- **Robust Generalization**: The model maintained high accuracy on unseen test data, demonstrating robustness.
- **Test Set Integrity**: By partitioning the test set before augmentation, we ensured that performance metrics accurately reflect the model's ability to generalize.

---

## Future Work

- **External Validation**: Evaluate the model on entirely separate datasets to further confirm robustness.
- **Advanced Augmentation Techniques**: Explore other augmentation methods, such as Variational Autoencoders (VAEs) or Diffusion Models, to enhance data diversity.
- **Clinical Trials**: Collaborate with medical professionals to assess the model's utility in clinical settings.

---

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- GPU (recommended for training DCGAN)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/LUNG-GENESIS.git
   cd LUNG-GENESIS
