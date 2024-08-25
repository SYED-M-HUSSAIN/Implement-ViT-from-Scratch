# Vision Transformer from Scratch

This repository contains an implementation of a Vision Transformer (ViT) research paper tiitle **"AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"** from scratch using PyTorch . The project is organized into separate modules for better readability and maintainability, following best practices.
![Vision Transformer Model](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/ViT.png)

## Project Structure

```
vision_transformer_project/
│
├── vision_transformer/
│   ├── __init__.py
│   ├── vision_transformer.py
│   ├── mlp_head.py
│   ├── transformer_encoder.py
│   ├── layer_norm.py
│   └── utils.py
│
├── main.py
└── requirements.txt
```

## Components
=======
### Vision Transformer (ViT)
An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings.


![Vision Transformer Model](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/ViT.png)


1. **Vision Transformer**: The main Vision Transformer class.
2. **MLP Head**: The Multi-Layer Perceptron head for classification.
3. **Transformer Encoder**: The Transformer Encoder layer.
4. **Normalization Layer**: The Transformer Normalization layer.
5. **Utils**: Utility functions for image processing and patch embedding.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch.git
    cd vision_transformer_project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your image file (e.g., `Image.png`) in the project directory.

2. Run the main script:
    ```bash
    python main.py
    ```

## Code Overview

### Vision Transformer

The Vision Transformer class is defined in `vision_transformer/vision_transformer.py`. It includes methods for patch embedding, positional encoding, and transformer encoder layers.

![Equations](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/equations.png)
### MLP Head

The MLP Head class is defined in `vision_transformer/mlp_head.py`. It is used for the final classification task.
![MLP](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/mlp.png)
### Transformer Encoder

The Transformer Encoder class is defined in `vision_transformer/transformer_encoder.py`. It includes multi-head attention and feed-forward layers.

![Encoder](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/encoder.png)

### Normalization Layer

The Transformer Layer Normalization class is defined in `vision_transformer/layer_norm.py`.

![Norm](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/norm.png)

### Utils

Utility functions for image processing and patch embedding are defined in `vision_transformer/utils.py`.
![TABLE](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/blob/main/content/table.png)

## Example

Here is an example of how to use the Vision Transformer:

```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- The implementation is inspired by the Vision Transformer (ViT) paper by Dosovitskiy et al, [Published as a conference paper at ICLR 2021](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/content/Research%20paper.pdf)

=======
### Equations

The equations used in our model are as follows:



1. **Initial Embedding**:
```math
   
    z_0 = [x_{class}; x_1^pE; x_2^pE; \cdots; x_N^pE] + E_{pos}, E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N+1) \times D}
   
```
2. **Multi-Head Self-Attention (MSA) with Layer Normalization (LN)**:
```math
   
   z_0^l = MSA(LN(z_{l-1})) + z_{l-1}, l = 1 \ldots L
```
3. **Multi-Layer Perceptron (MLP) with Layer Normalization (LN)**:
```math
   
   z_l = MLP(LN(z_0^l)) + z_0^l, l = 1 \ldots L
   
```
4. **Final Output**:
```math
   
   y = LN(z_0^L)
   
```

### References
[Published as a conference paper at ICLR 2021](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/content/Research%20paper.pdf)