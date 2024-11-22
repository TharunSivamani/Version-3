import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from mnist_model import LightMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

def show_augmented_images(num_images=5):
    """Display original and augmented versions of MNIST images."""
    # Basic transform (just ToTensor)
    basic_transform = transforms.ToTensor()
    
    # Advanced augmentation transform
    augment_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])
    
    # Load dataset without any transform
    dataset = datasets.MNIST('data', train=True, download=True, transform=None)
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # Get a random image (PIL Image)
        img, _ = dataset[np.random.randint(len(dataset))]
        
        # Convert to tensor for original image
        orig_tensor = basic_transform(img)
        
        # Show original
        axes[0, i].imshow(orig_tensor.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Apply augmentation (on PIL Image)
        aug_tensor = augment_transform(img)
        
        # Show augmented
        axes[1, i].imshow(aug_tensor.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

def test_model_confidence():
    """Test if model predictions are confident (high probability for correct class)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightMNIST().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    confidences = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            max_probs = probabilities.max(dim=1)[0]
            confidences.extend(max_probs.cpu().numpy())
    
    avg_confidence = np.mean(confidences)
    assert avg_confidence > 0.9, f"Average prediction confidence is {avg_confidence:.2f}, should be above 0.9"
    return avg_confidence

def test_model_invariance():
    """Test model's invariance to small translations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightMNIST().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    # Original transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Transform with small translation
    translate_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    translated_dataset = datasets.MNIST('data', train=False, download=True, transform=translate_transform)
    translated_loader = DataLoader(translated_dataset, batch_size=128, shuffle=False)
    
    def get_accuracy(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        return 100. * correct / total
    
    original_acc = get_accuracy(test_loader)
    translated_acc = get_accuracy(translated_loader)
    
    acc_diff = abs(original_acc - translated_acc)
    assert acc_diff < 5.0, f"Accuracy difference {acc_diff:.2f}% is too large (should be < 5%)"
    return acc_diff

def test_model_digit_consistency():
    """Test if model performs consistently across all digit classes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightMNIST().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Track accuracy for each digit
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            
            # Update accuracy for each class
            for i in range(len(target)):
                label = target[i]
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1
    
    # Calculate per-class accuracies
    class_accuracies = [100 * correct / total 
                       for correct, total in zip(class_correct, class_total)]
    
    # Calculate standard deviation of accuracies
    acc_std = np.std(class_accuracies)
    mean_acc = np.mean(class_accuracies)
    
    # Print per-class accuracies
    for i in range(10):
        print(f"Digit {i}: {class_accuracies[i]:.2f}%")
    print(f"\nMean accuracy: {mean_acc:.2f}%")
    print(f"Standard deviation: {acc_std:.2f}%")
    
    # Assert that standard deviation is not too high (model should be consistent)
    assert acc_std < 5.0, f"Accuracy variation between digits ({acc_std:.2f}%) is too high"
    # Assert that mean accuracy is good
    assert mean_acc > 90.0, f"Mean accuracy ({mean_acc:.2f}%) is too low"
    
    return mean_acc, acc_std

if __name__ == "__main__":
    # Generate augmented image samples
    print("Generating augmented image samples...")
    show_augmented_images()
    
    print("\nRunning confidence test...")
    conf = test_model_confidence()
    print(f"Average prediction confidence: {conf:.2f}")
    
    print("\nRunning invariance test...")
    diff = test_model_invariance()
    print(f"Translation invariance difference: {diff:.2f}%")
    
    print("\nRunning digit consistency test...")
    mean_acc, acc_std = test_model_digit_consistency()
    print(f"Consistency test passed with {acc_std:.2f}% standard deviation") 