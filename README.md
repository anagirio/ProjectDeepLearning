# Handwritten Digit Classification with CNNs and PyTorch

## Overview

This project explores handwritten digit classification using a Convolutional Neural Network (CNN) implemented in PyTorch.

The model is designed to recognize digits from image data, demonstrating a practical deep learning workflow involving preprocessing, neural network training, validation, and prediction generation.

## Technologies

- Python
- PyTorch
- Convolutional Neural Networks (CNNs)
- Computer Vision
- Deep Learning

## Model Architecture

The implementation includes:

- Three convolutional layers for feature extraction
- Batch normalization
- Dropout for regularization
- Fully connected layers for classification

## Training and Evaluation

The training workflow includes:

- Image preprocessing and data augmentation
- Model optimization using Adam
- Learning rate scheduling
- Validation accuracy monitoring
- Saving the best-performing model checkpoint

## Repository Structure

- `main.py` — Model implementation and training pipeline
- `best_model.pt` — Saved model checkpoint
- `train.csv` — Training data
- `test.csv` — Test data
- Submission CSV files — Generated model predictions

## Project Status

Academic deep learning project demonstrating CNN implementation, training, and evaluation with PyTorch.

## Future Improvements

- Improve separation of training and validation preprocessing
- Document reproducible training instructions and dependencies
- Add verified evaluation metrics and training visualizations
