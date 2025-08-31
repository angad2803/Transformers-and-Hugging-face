# Transformer Encoder Structure

D:\DL_Transformers_HuggingFace\transformer.png

A Transformer Encoder is basically a stack of identical blocks (BERT uses 12, DNABERT uses 6 for efficiency). Each block has 3 main parts:

Multi-Head Self-Attention

Feed-Forward Neural Network (FFN)

Add & Norm (Residual + LayerNorm)
