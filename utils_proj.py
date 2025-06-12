import torch
from dataset.dataloader import Dataset


def get_best_device() -> torch.device:
    """
    Returns the best available device for PyTorch.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device=device)

def show_datasets():
    dataset:Dataset = Dataset(
        path="dataset",
        condition="no_obj"
    )
    train_set = dataset.get_training_set()
    val_set = dataset.get_validation_set()
    print(f"Training set size: {len(train_set[0])}")
    print(f"Validation set size: {len(val_set[0])}")
    sample = dataset[0]
    print(f"Sample data shape: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")

device:torch.device = get_best_device()