import torchvision.transforms as transforms

# Common transformations for both training and testing
COMMON_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Advanced transformations for augmentation visualization
AUGMENTATION_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
])

# Test transforms (minimal for consistent evaluation)
TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Noise test transforms
NOISE_TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
])

# Translation test transforms
TRANSLATION_TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
]) 