The Problem- Attention has no sense of order.

Self-attention treats a sequence as a set → it only cares about relationships (Q·K similarities), not where tokens appear.

Example: “Bank near the river” vs “River near the bank”.

Same words, but meaning depends on word order.

So we need a way to inject position information into embeddings.
