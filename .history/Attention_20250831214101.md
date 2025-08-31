# What is Attention

Attention is a mechanism for focusing on relevant parts of the input when making predictions.
Instead of treating all input tokens equally, attention asks:

👉 “When I process this word (or token), which other words are most important for understanding it?”

## The core math behind attention is Scaled dot product

We have three matrices

Queries (Q): What I’m looking for (like a question).

Keys (K): Labels that describe what each token has (like tags).

Values (V): The actual content to be retrieved.

![alt text](image.png)

## Self Attention

Special case of attention where:

Q, K, V are all derived from the same sequence.

That’s why it’s called self attention → each token decides how much to pay attention to other tokens in the same sequence.

Example:
Sentence = “The cat sat on the mat.”

When encoding “cat”, the model may attend strongly to “sat” and “mat” (useful context).

“the” will attend less since it’s less informative.

So self-attention = contextualizing each token with respect to the whole sequence.

**_ Contextual embedding is the core of self attention as for example Money bank and River Bank , the Bank here is diff and the model needs to know of how to differentiate them , or the context behind them, That is done by Attention _**

## Multi-Head Attention

![alt text](image-1.png)

self explanatory by the name , crave for different data and relationships within the data

# Summary

Attention = weighted focus mechanism using queries, keys, values.

Self-Attention = Q=K=V from the same sequence → token attends to itself + others.

Scaled Dot-Product = stabilizes learning by normalizing dot products.

Multi-Head Attention = multiple attention “perspectives” → richer context.

![alt text](image-2.png)
