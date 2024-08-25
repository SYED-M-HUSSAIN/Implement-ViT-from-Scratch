import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .mlp_head import MLPHead

class VisionTransformer(nn.Module):
    def __init__(self, patch_size: int = 16, image_size: int = 224, channel_size: int = 3, num_layers: int = 12,
                 embedding_dim: int = 768, num_heads: int = 12, hidden_dim: int = 3072, dropout_prob: float = 0.1,
                 num_classes: int = 10, pretrain: bool = True):
        """
        Vision Transformer (ViT) implementation.

        Args:
            patch_size (int): Size of each image patch.
            image_size (int): Size of the input image.
            channel_size (int): Number of channels in the input image.
            num_layers (int): Number of transformer encoder layers.
            embedding_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the hidden layer in the feed-forward network.
            dropout_prob (float): Dropout probability.
            num_classes (int): Number of output classes.
            pretrain (bool): Whether to use pre-trained weights.
        """
        super().__init__()
        self.patch_size = patch_size
        self.channel_size = channel_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_size * patch_size * channel_size, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches + 1, embedding_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_layers)]
        )
        self.mlp_head = MLPHead(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_size, channel_size = self.patch_size, self.channel_size
        patches = x.unfold(1, channel_size, channel_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(patches.size(0), -1, channel_size * patch_size * patch_size).float()

        patch_embeddings = torch.matmul(patches, self.patch_embedding_weight)

        batch_size = patch_embeddings.shape[0]
        patch_embeddings = torch.cat((self.class_token.repeat(batch_size, 1, 1), patch_embeddings), dim=1)

        patch_embeddings += self.position_embedding
        transformer_output = self.transformer_encoders(patch_embeddings)
        class_token_output = transformer_output[:, 0]
        output = self.mlp_head(class_token_output)
        return output