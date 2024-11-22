import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from mnist_model import LightMNIST, train_one_epoch

def test_saved_model():
    """Load and test the saved model for one epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and load saved weights
    model = LightMNIST()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Use the same transform as in training
    transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset
    print("\nLoading MNIST test dataset...")
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Test the model
    correct = 0
    total = 0
    
    print("\nTesting saved model...")
    with torch.no_grad():  # No need to track gradients
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            accuracy = 100. * correct / total
            pbar.set_postfix({'acc': f'{accuracy:.2f}%'})
    
    final_accuracy = 100. * correct / total
    print(f"\nTest accuracy: {final_accuracy:.2f}%")
    
    # Assert accuracy is above 95%
    assert final_accuracy >= 95.0, f"Model accuracy is {final_accuracy:.2f}%, should be at least 95%"
    
    return final_accuracy

if __name__ == "__main__":
    test_saved_model() 