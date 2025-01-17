# Image Classification CNN Model

## Overview
This repository contains an implementation of a Convolutional Neural Network (CNN) for image classification using TensorFlow/Keras. The model features a sophisticated architecture with skip connections, batch normalization, and comprehensive regularization techniques.

## Model Architecture
- **Input Layer**: Accepts RGB images of size 128x128
- **Backbone**: 
  - 4 convolutional blocks with increasing filters (32 → 64 → 128 → 256)
  - Skip connections for improved gradient flow
  - Batch normalization and dropout layers for regularization
- **Head**:
  - Global average pooling
  - Dense layers with skip connections
  - Softmax output layer for classification

## Features
- Data augmentation pipeline
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Checkpoint saving for best model weights
- Weight decay regularization
- Stratified train-validation split (90-10)

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

## Project Structure
```
.
├── images/
│   └── classes/    # Image data organized by class
├── notebooks/
│   └── notebooks.ipynb/    # Image data organized by class
```

## Configuration
The model uses an optimized configuration dictionary with the following parameters:
```python
OPTIMIZED_CONFIG = {
    'data_path': '/path/to/dataset',
    'image_size': (128, 128),
    'batch_size': 64,
    'epochs': 50,
    'initial_learning_rate': 0.0001,
    'num_classes': 10,
    'train_split': 0.9,
    'weight_decay': 0.0005,
    'dropout_rate': 0.3,
    'early_stopping_patience': 10
}
```

## Usage

1. **Prepare Your Dataset**
   - Organize your images in class-specific folders under the main dataset directory
   - Update the `data_path` in the configuration

2. **Training**
   ```python
   python main.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train the model with the specified configuration
   - Save the best model weights
   - Generate training history plots

3. **Monitoring**
   - Training progress is displayed in real-time
   - Training/validation accuracy and loss plots are generated automatically
   - Early stopping monitors validation loss with a patience of 10 epochs

## Model Features

### Data Processing
- Automatic train-validation split (90-10)
- Image rescaling and normalization
- Optional data augmentation (currently commented out):
  - Rotation
  - Width/height shifts
  - Zoom
  - Horizontal flips

### Training Optimizations
- Gradient clipping for stability
- Learning rate reduction on plateau
- L2 regularization on convolutional and dense layers
- Progressive dropout rates (0.25 → 0.4)
- Skip connections for better gradient flow

## Output
- Training metrics (accuracy and loss) are displayed during training
- Best model weights are saved to 'best_model.keras'
- Training history plots are generated showing accuracy and loss curves

## Error Handling
The implementation includes comprehensive error handling for:
- Invalid data paths
- Empty datasets
- Training interruptions
- Memory issues

## Notes
- The model is designed for RGB images of size 128x128
- Batch size is optimized for common GPU memory constraints
- Early stopping prevents overfitting while ensuring optimal model performance
