# Lightweight MNIST Model

This repository contains a lightweight MNIST model with the following characteristics:
- Less than 25,000 parameters
- Achieves >95% training accuracy in 1 epoch
- Fully tested with GitHub Actions

## Model Architecture
The model uses an efficient CNN architecture:
- Input Block: 1 channel → 8 channels (3x3 conv)
- Convolution Block 1: 8 → 16 channels (3x3 conv)
- Transition Block: 16 → 8 channels (1x1 conv)
- Convolution Block 2: Multiple layers (8 → 16 → 8 → 16 → 16 channels)
- Output Block: Global Average Pooling → 10 classes

Key features:
- Uses 1x1 convolutions for channel reduction
- Employs Global Average Pooling
- Includes BatchNorm and Dropout
- No bias in convolutional layers

## Requirements
```bash
pip install -r requirements.txt
```

## Running the Model

1. Train the model:
```bash
python mnist_model.py
```

2. Test the saved model:
```bash
python test_saved_model.py
```

3. Run advanced tests:
```bash
python advanced.py
```

## Testing Framework

### Basic Tests
- Parameter count verification (< 25k)
- Training accuracy check (> 95%)
- Model architecture validation

### Advanced Tests
1. Robustness Test:
   - Performance on noisy images
   - Minimum 85% accuracy with noise

2. Confidence Test:
   - Prediction confidence analysis
   - Average confidence > 0.9

3. Translation Invariance Test:
   - Performance with image translations
   - Maximum 5% accuracy drop

4. Digit Consistency Test:
   - Per-digit accuracy analysis
   - Balanced performance across all digits

## CI/CD Pipeline
The GitHub Actions workflow:
1. Pre-test parameter check
2. Model training and validation
3. Advanced testing suite
4. Artifact collection (augmented samples)

## Project Structure
```
├── mnist_model.py        # Main model and training
├── test_model.py         # Basic tests
├── test_saved_model.py   # Saved model verification
├── advanced.py           # Advanced testing suite
├── pre_test.py          # Parameter verification
├── transforms_config.py  # Data augmentation setup
├── requirements.txt      # CPU-only dependencies
└── .github/workflows     # CI/CD configuration
```

## Data Augmentation
The model uses several augmentation techniques:
- Random rotation (±15°)
- Random affine transformations
- Random perspective
- Random erasing

## Model Performance
- Parameter Count: < 25,000
- Training Accuracy: > 95% (1 epoch)
- Test Accuracy: > 90%
- Robust to noise and translations

