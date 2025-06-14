import torch
from dataset.dataloader import Dataset
from PIL import Image

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

def image_list_to_gif(image_list, filename, duration=100):
    """
    Save a list of images as a GIF.
    Args:
        image_list (list): List of images torch tensors (3, 64, 64).
        filename (str): Output filename for the GIF.
        duration (int): Duration for each frame in milliseconds.
    """
    if not image_list:
        raise ValueError("The image list is empty.")
    
    # Convert tensors to PIL images
    pil_images = []
    for img_tensor in image_list:
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu().numpy()  # Move to CPU and convert to numpy
        if img_tensor.ndim == 3:  # Ensure the image is in (C, H, W) format
            img_tensor = img_tensor.transpose(1, 2, 0)  # Convert to (H, W, C)
        pil_image = Image.fromarray((img_tensor * 255).astype('uint8'))  # Convert to PIL Image
        pil_images.append(pil_image)
    if not pil_images:
        raise ValueError("No valid images to save as GIF.")

    image_list = pil_images
    image_list[0].save(filename, save_all=True, append_images=image_list[1:], duration=duration, loop=0)
    print(f"GIF saved as {filename}")