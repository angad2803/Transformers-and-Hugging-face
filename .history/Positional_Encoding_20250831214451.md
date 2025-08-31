The Problem- Attention has no sense of order.

Self-attention treats a sequence as a set → it only cares about relationships (Q·K similarities), not where tokens appear.

Example: “Bank near the river” vs “River near the bank”.

Same words, but meaning depends on word order.

So we need a way to inject position information into embeddings.

![alt text](image-3.png)

## What "Attention is All You Need" Did

They introduced sinusoidal positional encoding added to word embeddings.

For each position pos and dimension i:

𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
)
=
sin
⁡
(
𝑝
𝑜
𝑠
10000
2
𝑖
/
𝑑
𝑚
𝑜
𝑑
𝑒
𝑙
)
PE
(pos,2i)
​

=sin(
10000
2i/d
model
​

pos
​

)
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖

- 1
  )
  =
  cos
  ⁡
  (
  𝑝
  𝑜
  𝑠
  10000
  2
  𝑖
  /
  𝑑
  𝑚
  𝑜
  𝑑
  𝑒
  𝑙
  )
  PE
  (pos,2i+1)
  ​

=cos(
10000
2i/d
model
​

pos
​

)

This way:

Low-frequency sinusoids capture global order (long-range).

High-frequency sinusoids capture local order (nearby words).

Since sine/cosine are periodic, the encoding generalizes to unseen sequence lengths.
