import torch
from mnist_model import LightMNIST

def check_model_params():
    """
    Check if model has less than 25000 parameters before running tests.
    Returns:
        bool: True if model passes parameter check
    """
    try:
        # Initialize model
        model = LightMNIST()
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Parameter Count Check:")
        print("=" * 50)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Parameter limit: 25,000")
        print("-" * 50)
        
        # Check if parameters are within limit
        if total_params >= 25000:
            print("❌ FAILED: Model has too many parameters!")
            print(f"    Expected < 25,000 but got {total_params:,}")
            return False
            
        print("✓ PASSED: Model parameters within limit")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to check model parameters")
        print(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_model_params()
    if not success:
        exit(1)  # Exit with error code if check fails 