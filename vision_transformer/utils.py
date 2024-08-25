import numpy as np
import torch
from PIL import Image

def image_to_patches(image_path: str, patch_size: int = 16, image_size: int = 224) -> torch.Tensor:
    """
    Convert an image to patches.

    Args:
        image_path (str): Path to the image file.
        patch_size (int): Size of each patch.
        image_size (int): Size to which the image is resized.

    Returns:
        torch.Tensor: Tensor containing image patches.
    """
    image = Image.open(image_path).resize((image_size, image_size))
    image_array = np.array(image)
    num_channels = 3
    patches = image_array.reshape(image_array.shape[0] // patch_size, patch_size, 
                                  image_array.shape[1] // patch_size, patch_size, num_channels)
    patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, num_channels)
    patches_flat = patches.reshape(-1, patch_size * patch_size * num_channels)
    return torch.tensor(patches_flat, dtype=torch.float32)

def get_patch_embeddings(patches: torch.Tensor, embedding_dim: int = 768) -> torch.Tensor:
    """
    Get patch embeddings.

    Args:
        patches (torch.Tensor): Tensor containing image patches.
        embedding_dim (int): Dimension of the embedding.

    Returns:
        torch.Tensor: Tensor containing patch embeddings.
    """
    patch_size = 16
    num_channels = 3
    patches = patches.unsqueeze(0)
    embedding_weights = torch.randn(1, patch_size * patch_size * num_channels, embedding_dim)
    patch_embeddings = torch.matmul(patches, embedding_weights)
    return patch_embeddings