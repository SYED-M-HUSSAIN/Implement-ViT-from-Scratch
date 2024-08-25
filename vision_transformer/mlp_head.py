import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_classes: int = 10, fine_tune: bool = False):
        """
        Multi-Layer Perceptron (MLP) head for the Vision Transformer.

        Args:
            embedding_dim (int): Dimension of the embedding.
            num_classes (int): Number of output classes.
            fine_tune (bool): Whether to fine-tune the model.
        """
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        self.mlp_head = self._build_mlp_head(embedding_dim, num_classes, fine_tune)

    def _build_mlp_head(self, embedding_dim: int, num_classes: int, fine_tune: bool) -> nn.Module:
        """
        Build the MLP head.

        Args:
            embedding_dim (int): Dimension of the embedding.
            num_classes (int): Number of output classes.
            fine_tune (bool): Whether to fine-tune the model.

        Returns:
            nn.Module: The MLP head module.
        """
        if fine_tune:
            return nn.Linear(embedding_dim, num_classes)
        else:
            return nn.Sequential(
                nn.Linear(embedding_dim, 3072),
                nn.Tanh(),
                nn.Linear(3072, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.mlp_head(x)