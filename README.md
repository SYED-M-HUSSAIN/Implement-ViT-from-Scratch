# Implement-ViT-from-Scratch

## AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

### Abstract
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

### Vision Transformer (ViT)
An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image \( x \in \mathbb{R}^{H \times W \times C} \) into a sequence of flattened 2D patches \( x_p \in \mathbb{R}^{N \times (P^2 \cdot C)} \), where \( (H, W) \) is the resolution of the original image, \( C \) is the number of channels, \( (P, P) \) is the resolution of each image patch, and \( N = \frac{HW}{P^2} \) is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size \( D \) through all of its layers, so we flatten the patches and map to \( D \) dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.

![Vision Transformer Model](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/content/ViT.png)

Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches (\( z_0^0 = x_{class} \)), whose state at the output of the Transformer encoder (\( z_0^L \)) serves as the image representation \( y \) (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to \( z_0^L \). The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.

Position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings. The resulting sequence of embedding vectors serves as input to the encoder.

The Transformer encoder consists of alternating layers of multiheaded self-attention (MSA) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block. The MLP contains two layers with a GELU non-linearity.

### Equations
\[ z_0 = [x_{class}; x_1^pE; x_2^pE; \cdots; x_N^pE] + E_{pos}, E \in \mathbb{R}^{(P^2 \cdot C) \times D}, E_{pos} \in \mathbb{R}^{(N+1) \times D} \]

\[ z_0^l = MSA(LN(z_{l-1})) + z_{l-1}, l = 1 \ldots L \]

\[ z_l = MLP(LN(z_0^l)) + z_0^l, l = 1 \ldots L \]

\[ y = LN(z_0^L) \]

### References
[Published as a conference paper at ICLR 2021](https://github.com/SYED-M-HUSSAIN/Implement-ViT-from-Scratch/content/Research%20paper.pdf)
