# ========================================
# DNA-BERT Fine-Tuning Pipeline
# ========================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd

# ----------------------------
# 1. Load your dataset
# ----------------------------
# Expected format: CSV with two columns: "sequence", "label"
# Example: sequence = "ATGCGTACG...", label = "Protist"
data = pd.read_csv("edna_sequences.csv")

# Encode labels as integers
labels = sorted(data["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}
data["label_id"] = data["label"].map(label2id)

# Train/valid/test split
train_df, temp_df = train_test_split(data, test_size=0.3, stratify=data["label_id"], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

# Convert to Hugging Face datasets
train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
test_ds  = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_ds,
    "validation": valid_ds,
    "test": test_ds
})

# ----------------------------
# 2. Preprocessing (k-mers)
# ----------------------------
def kmers(seq, k=6):
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])

# Load pretrained DNA-BERT tokenizer
MODEL = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)

def preprocess(example):
    example["sequence"] = kmers(example["sequence"])
    tokens = tokenizer(example["sequence"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = example["label_id"]
    return tokens

encoded_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# ----------------------------
# 3. Load DNA-BERT model
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ----------------------------
# 4. Define metrics
# ----------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"]
    }

# ----------------------------
# 5. Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    report_to="none"  # disable wandb if you don’t use it
)

# ----------------------------
# 6. Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ----------------------------
# 7. Train
# ----------------------------
trainer.train()

# ----------------------------
# 8. Evaluate
# ----------------------------
metrics = trainer.evaluate(encoded_dataset["test"])
print("Test Metrics:", metrics)
