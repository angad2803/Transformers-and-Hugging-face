# BERT: Full Explanation (Devlin et al., 2018)

BERT uses an encoder-only Transformer architecture focused on understanding text.

## Encoder Block Structure

- Multi-head self-attention
- Feedforward neural network (FFNN)
- Add & Norm (residual connections)

**Why encoder-only?**

- Decoders (for generation) are not needed; BERT is for comprehension.
- Bidirectional self-attention enables rich representation learning.

BERT (Bidirectional Encoder Representation from Transformers) is widely fine-tuned for many NLP tasks.

![BERT architecture](image-4.png)

## Input Embedding

Each token’s input embedding is the sum of:

- **Token Embedding:** Learned for each word-piece (WordPiece tokenizer)
- **Segment Embedding:** Distinguishes sentence A from B (for NSP)
- **Position Embedding:** Adds positional info (since Transformers lack recurrence)

`InputEmbedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding`

![Input embedding](image-5.png)

## Pretraining Tasks

### 1. Masked Language Modeling (MLM)

**Goal:** Predict randomly masked tokens using bidirectional context.

- Randomly mask 15% of tokens:
  - 80% replaced with `[MASK]`
  - 10% replaced with a random token
  - 10% unchanged
- Model predicts the original token at each masked position.

MLM enables BERT to learn nuanced meanings from both directions.

### 2. Next Sentence Prediction (NSP)

**Goal:** Predict if sentence B follows sentence A.

- 50%: B is the true next sentence
- 50%: B is a random sentence
- Model predicts IsNext / NotNext

NSP teaches BERT about sentence-level coherence, useful for QA, NLI, and entailment.

✅ Pretraining Output: Deep bidirectional contextual embeddings.

## Fine-tuning

After pretraining, BERT is adapted for supervised NLP tasks by adding a small output layer.

- Input representation: same as pretraining
- Entire BERT is fine-tuned with labeled data
- Only the output layer changes per task

**Examples:**

- **Single sentence classification (e.g., sentiment):**
  - Input: `[CLS] sentence [SEP]`
  - Output: `[CLS]` embedding → softmax classifier
- **Sentence pair tasks (e.g., NLI, QA):**
  - Input: `[CLS] sentence_A [SEP] sentence_B [SEP]`
  - Output: `[CLS]` embedding → softmax for entailment/similarity
- **Token-level tasks (e.g., NER, POS):**
  - Input: `[CLS] sentence [SEP]`
  - Output: Each token embedding → classification head
- **Question Answering (e.g., SQuAD):**
  - Input: `[CLS] question [SEP] passage [SEP]`
  - Output: Predict start/end positions of answer span

## Training Passes

**Pretraining (MLM):**

- Input: “The cat sat on the [MASK].”
- BERT computes contextual embeddings.
- At `[MASK]`, output hidden state → classification layer → softmax over vocab.
- Loss: cross-entropy with true word.

**Pretraining (NSP):**

- Input: `[CLS] The cat sat on the mat. [SEP] It was tired. [SEP]`
- `[CLS]` embedding → linear layer → sigmoid (IsNext/NotNext)
- Loss: binary cross-entropy.

**Fine-tuning (e.g., sentiment):**

- Input: `[CLS] I loved the movie. [SEP]`
- `[CLS]` embedding → classification layer → softmax (Positive/Negative)
- Loss: cross-entropy.

![alt text](image-6.png)
