import torch
import torch.nn.functional as F
import torchvision.transforms as T
from vision_transformer import VisionTransformer, image_to_patches, get_patch_embeddings

def main():
    IMAGE_SIZE = 224
    CHANNEL_SIZE = 3
    NUM_CLASSES = 10
    DROPOUT_PROB = 0.1
    NUM_LAYERS = 12
    EMBEDDING_DIM = 768
    NUM_HEADS = 12
    HIDDEN_DIM = 3072
    PATCH_SIZE = 16
    IMAGE_PATH = 'Image.png'

    image_patches = image_to_patches(IMAGE_PATH, PATCH_SIZE)
    patch_embeddings = get_patch_embeddings(image_patches, EMBEDDING_DIM)

    vision_transformer = VisionTransformer(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        channel_size=CHANNEL_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        dropout_prob=DROPOUT_PROB,
        num_classes=NUM_CLASSES
    )

    vit_output = vision_transformer(patch_embeddings)
    print(vit_output.shape)

    probabilities = F.softmax(vit_output[0], dim=0)
    print(probabilities)
    print(torch.sum(probabilities))

if __name__ == "__main__":
    main()