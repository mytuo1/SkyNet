# Run this OUTSIDE Docker on your machine or a training server
# Uses a small BERT-family model — runs on CPU/GPU with 8GB RAM

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report

# ── Step 1: Load GoEmotions ────────────────────────────────────────────────────

dataset = load_dataset("go_emotions", "simplified")
# simplified = 28 labels (27 emotions + neutral)

# Label mapping
LABELS = dataset["train"].features["labels"].feature.names
NUM_LABELS = len(LABELS)   # 28
print(f"Training on {NUM_LABELS} emotion labels: {LABELS}")


# ── Step 2: Choose base model ──────────────────────────────────────────────────
# Options by resource requirement:
#   roberta-base         → best quality, needs 8GB RAM, ~125M params
#   distilbert-base      → good quality, 4GB RAM, ~66M params  
#   bert-tiny            → runs on 2GB, weaker but fast
#   microsoft/deberta-v3-small → excellent for emotion, 8GB

MODEL_NAME = "distilbert-base-uncased"   # change to roberta-base if you have resources
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Step 3: Preprocess ─────────────────────────────────────────────────────────

def preprocess(examples):
    """Tokenize and convert multi-label lists to multi-hot vectors."""
    tokenized = tokenizer(
        examples["text"],
        max_length  = 128,
        truncation  = True,
        padding     = "max_length"
    )
    
    # Convert list of label indices to multi-hot vector
    labels_matrix = np.zeros((len(examples["labels"]), NUM_LABELS))
    for i, label_list in enumerate(examples["labels"]):
        for label_idx in label_list:
            labels_matrix[i][label_idx] = 1.0
    
    tokenized["labels"] = labels_matrix.tolist()
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset.set_format("torch")


# ── Step 4: Model ─────────────────────────────────────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels            = NUM_LABELS,
    problem_type          = "multi_label_classification",
    id2label              = {i: l for i, l in enumerate(LABELS)},
    label2id              = {l: i for i, l in enumerate(LABELS)}
)


# ── Step 5: Metrics ───────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Sigmoid threshold at 0.5 for multi-label
    probs      = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probs > 0.5).astype(int)
    
    # Micro F1 is the standard metric for GoEmotions
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


# ── Step 6: Training ──────────────────────────────────────────────────────────

args = TrainingArguments(
    output_dir                  = "./emotion_model",
    num_train_epochs            = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "micro_f1",
    fp16                        = torch.cuda.is_available(),
    logging_steps               = 100,
    report_to                   = "none"
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = tokenized_dataset["train"],
    eval_dataset    = tokenized_dataset["validation"],
    compute_metrics = compute_metrics
)

trainer.train()
trainer.save_model("./emotion_model/final")
tokenizer.save_pretrained("./emotion_model/final")

# Expected results after 5 epochs on distilbert:
# Micro F1: ~0.52–0.56 (GoEmotions is genuinely hard)
# Macro F1: ~0.45–0.50
# For context, human agreement on GoEmotions is ~0.54 Macro F1
print("Training complete. Model saved to ./emotion_model/final")
