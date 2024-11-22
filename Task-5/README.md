# Lightweight MNIST Model

This repository contains a lightweight MNIST model with the following characteristics:
- Less than 25,000 parameters
- Achieves >95% training accuracy in 1 epoch
- Fully tested with GitHub Actions

## Model Architecture
The model uses a simple CNN architecture with:
- 2 convolutional layers
- 2 max pooling layers
- 1 fully connected layer

## Requirements
```
pip install -r requirements.txt
```

## Running the Model
```
python mnist_model.py
```

## Running Tests
```
pytest test_model.py -v
```

## Implementation Details
- Creates a lightweight CNN model with approximately 8,000 parameters
- Uses efficient architecture with minimal layers but effective feature extraction
- Implements GitHub Actions testing for both parameter count and accuracy requirements
- Includes proper testing infrastructure with pytest
- Provides complete documentation

## Usage
1. Create a new GitHub repository
2. Push all these files to the repository
3. GitHub Actions will automatically run the tests on push and pull requests

## Model Architecture Details
- First conv layer: 8 filters (reduces parameters)
- Second conv layer: 16 filters
- Single fully connected layer at the end
- Uses max pooling to reduce spatial dimensions
- ReLU activation for non-linearity

## CI/CD Pipeline
The GitHub Actions workflow will:
1. Set up a Python environment
2. Install dependencies
3. Run the tests to verify both requirements (parameter count and accuracy)

