import torch
from dataset.dataloader import Dataset
from PIL import Image
import matplotlib.pyplot as plt

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
    # handle shape 64 64 3 by rechaping
    if isinstance(image_list[0], torch.Tensor):
        if image_list[0].shape == (64, 64, 3):
            print("Reshaping image tensors to (C, H, W) format.")
            for i in range(len(image_list)):
                image_list[i] = image_list[i].permute(2, 0, 1)
    
    # Convert tensors to PIL images
    pil_images = []
    for img_tensor in image_list:
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu().numpy()  # Move to CPU and convert to numpy
        if img_tensor.ndim == 3:  # Ensure the image is in (C, H, W) format
            img_tensor = img_tensor.transpose(1, 2, 0)  # Convert to (H, W, C)
        pil_image = Image.fromarray((img_tensor.clip(0, 1) * 255).astype('uint8'))  # Clip, scale, and convert to PIL Image
        pil_images.append(pil_image)
    if not pil_images:
        raise ValueError("No valid images to save as GIF.")

    image_list = pil_images
    image_list[0].save(filename, save_all=True, append_images=image_list[1:], duration=duration, loop=0)
    print(f"GIF saved as {filename}")

def show_image(image_tensor, title=None, save_path=None):
    """
    Display a single image tensor.
    Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
    """
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.cpu().numpy()  # Move to CPU and convert to numpy
    if image_tensor.ndim == 3:  # Ensure the image is in (C, H, W) format
        image_tensor = image_tensor.transpose(1, 2, 0)  # Convert to (H, W, C)
    plt.imshow(image_tensor)
    plt.axis('off')  # Hide axes
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Image saved as {save_path}")
    plt.show()

def save_image(image_tensor, filename):
    """
    Save a single image tensor to a file.
    Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
        filename (str): Output filename for the image.
    """
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.cpu().numpy()  # Move to CPU and convert to numpy
    if image_tensor.ndim == 3:  # Ensure the image is in (C, H, W) format
        image_tensor = image_tensor.transpose(1, 2, 0)  # Convert to (H, W, C)
    pil_image = Image.fromarray(image_tensor.clip(0, 1) * 255).astype('uint8')  # Clip to [0,1], scale, convert to PIL Image
    pil_image.save(filename)
    print(f"Image saved as {filename}")

def plot_loss(history: dict, title: str = 'Loss Curve', xlabel: str = 'Epochs', ylabel: str = 'Loss', save_path: str = None):
    '''
    Plot the training and validation loss.
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss'.
    Returns:
        None
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved as {save_path}")
    else:
        print("Loss plot not saved, no save path provided.")
    plt.legend()
    plt.show()