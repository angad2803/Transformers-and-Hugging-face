# Transformer Encoder Structure

D:\DL_Transformers_HuggingFace\transformer.png

A Transformer Encoder is basically a stack of identical blocks (BERT uses 12, DNABERT uses 6 for efficiency). Each block has 3 main parts:

Multi-Head Self-Attention

Feed-Forward Neural Network (FFN)

Add & Norm (Residual + LayerNorm)

1. Input → Embeddings + Positional Encoding

Input = sequence of tokens (DNA k-mers in DNABERT).

Each token is converted to a vector (embedding).

Since Transformers have no recurrence or convolution, we add positional encoding to inject order information.
