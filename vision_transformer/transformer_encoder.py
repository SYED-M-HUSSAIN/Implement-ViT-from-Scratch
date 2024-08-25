import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_attention_heads: int, feedforward_dim: int, dropout_rate: float):
        """
        Transformer Encoder layer.

        Args:
            embedding_dim (int): Dimension of the embedding.
            num_attention_heads (int): Number of attention heads.
            feedforward_dim (int): Dimension of the hidden layer in the feed-forward network.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_attention_heads, dropout=dropout_rate)
        self.feedforward_network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GeLU(),
            nn.Linear(feedforward_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.multihead_attention(input_tensor, input_tensor, input_tensor)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm1(input_tensor + attention_output)
        
        feedforward_output = self.feedforward_network(attention_output)
        feedforward_output = self.dropout(feedforward_output)
        output_tensor = self.layer_norm2(attention_output + feedforward_output)
        
        return output_tensor