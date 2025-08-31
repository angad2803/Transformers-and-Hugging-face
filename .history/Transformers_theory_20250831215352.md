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

2. Multi-Head Self-Attention

Takes Q, K, V from the same input sequence (self-attention).

Each “head” can focus on different relationships:

Head 1: promoter → gene

Head 2: motif → splice site

Head 3: long-range dependencies

⚡ Why Multi-Head? → If you only had one head, you’d capture only one type of relationship. Multiple heads = multiple perspectives.

3. Residual + Layer Normalization (Add & Norm)

Output of attention is added back to the input (residual connection).

This helps gradient flow and prevents vanishing gradients.

Think: “don’t throw away original info, just enrich it.”

Then LayerNorm is applied for stability (normalizes across features).

4. Feed Forward Network (FFN)

Each token embedding goes through a position-wise MLP (two linear layers with ReLU/GELU).

Same FFN is applied to each token independently.

Purpose: add non-linearity and let the model learn more abstract features (like motifs → biological meaning).

5. Residual + Layer Normalization (Again)

Same idea as before: output of FFN is added back to its input (residual) and normalized.

🔹 6. Stack of N Encoders (DNABERT → 6 Blocks)

Each block repeats (Self-Attention → AddNorm → FFN → AddNorm).

Why 6?

Deeper = more expressive, but also more expensive.

For DNA, long sequences mean very high memory cost.

6 layers = sweet spot (balance between performance & GPU feasibility).

Input Embedding + Positional Encoding
↓
Multi-Head Self Attention
↓ (Add & Norm)
Feed-Forward Network
↓ (Add & Norm)
Output → next encoder
