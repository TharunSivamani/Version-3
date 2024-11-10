import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import requests
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import time

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Initialize model, loss, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs=10):
    train_losses = []
    last_update_time = 0
    update_interval = 0.5  # Update every 0.5 seconds
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                avg_loss = running_loss / (batch_idx + 1)
                train_losses.append(avg_loss)
                
                # Send data to visualization server
                try:
                    requests.post('http://localhost:5000/update', 
                                json={'loss': avg_loss, 
                                     'epoch': epoch,
                                     'batch': batch_idx})
                except:
                    print("Couldn't connect to visualization server")
                
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}')
                last_update_time = current_time

    # Send final update
    try:
        requests.post('http://localhost:5000/update', 
                     json={'loss': avg_loss, 
                          'epoch': epochs-1,
                          'batch': batch_idx})
    except:
        print("Couldn't connect to visualization server")

def evaluate_random_samples():
    model.eval()
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    
    # Select 10 random indices
    random_indices = np.random.choice(len(images), 10, replace=False)
    
    plt.figure(figsize=(20, 4))
    for idx, i in enumerate(random_indices):
        image = images[i].to(device)
        true_label = labels[i].item()
        
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred_label = output.argmax(dim=1).item()
        
        plt.subplot(2, 5, idx + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    
    plt.savefig('static/results.png')
    plt.close()

if __name__ == "__main__":
    train(epochs=10)
    time.sleep(1)  # Give time for final update
    evaluate_random_samples() 