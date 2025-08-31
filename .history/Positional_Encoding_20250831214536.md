The Problem- Attention has no sense of order.

Self-attention treats a sequence as a set → it only cares about relationships (Q·K similarities), not where tokens appear.

Example: “Bank near the river” vs “River near the bank”.

Same words, but meaning depends on word order.

So we need a way to inject position information into embeddings.

![alt text](image-3.png)

## What "Attention is All You Need" Did

They introduced sinusoidal positional encoding added to word embeddings.

For each position `pos` and dimension `i`, the positional encoding is defined as:

```math
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

- **Low-frequency sinusoids** capture global order (long-range dependencies).
- **High-frequency sinusoids** capture local order (nearby words).

Because sine and cosine are periodic, these encodings generalize to unseen sequence lengths.

The positional encoding is added to embeddings before computing Q, K, V.

Now when queries and keys are compared (Q·Kᵀ), their similarity also reflects relative position.

Through linear transformations (W_Q, W_K, W_V), the model can form relations like:

“this word attends more strongly to the one 2 steps before me”

or “attend to the closest noun regardless of distance.”
