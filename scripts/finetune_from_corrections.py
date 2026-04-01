# Run periodically (nightly) to incorporate user corrections
# This IS the continual learning for emotion classification

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

CORRECTION_LOG = "/workspace/emotion_corrections.jsonl"
MODEL_PATH     = "./emotion_model/final"

def load_corrections() -> list:
    """
    Load corrections logged by the amygdala service.
    Format: {"text": "...", "predicted": [...], "correct": [...], "timestamp": ...}
    """
    corrections = []
    if Path(CORRECTION_LOG).exists():
        with open(CORRECTION_LOG) as f:
            for line in f:
                corrections.append(json.loads(line))
    return corrections


def incremental_finetune(corrections: list, base_dataset_fraction: float = 0.1):
    """
    Fine-tune on corrections + a fraction of original data.
    The fraction of original data prevents catastrophic forgetting
    — this is experience replay, a proven anti-forgetting technique.
    """
    if len(corrections) < 20:
        print(f"Only {len(corrections)} corrections — not enough to train yet")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    LABELS    = list(model.config.id2label.values())
    NUM_LABELS = len(LABELS)
    label2id  = {l: i for i, l in enumerate(LABELS)}
    
    # Build correction dataset
    correction_data = []
    for c in corrections:
        label_vec = np.zeros(NUM_LABELS)
        for emotion in c["correct"]:
            if emotion in label2id:
                label_vec[label2id[emotion]] = 1.0
        correction_data.append({
            "text":   c["text"],
            "labels": label_vec.tolist()
        })
    
    # Load a replay fraction from original GoEmotions
    # THIS IS THE KEY: replaying old examples prevents forgetting
    from datasets import load_dataset
    original = load_dataset("go_emotions", "simplified")["train"]
    replay_size = int(len(corrections) * 3)  # 3:1 ratio: old:new
    replay_indices = np.random.choice(len(original), min(replay_size, len(original)), replace=False)
    
    replay_data = []
    for idx in replay_indices:
        example   = original[int(idx)]
        label_vec = np.zeros(NUM_LABELS)
        for label_idx in example["labels"]:
            label_vec[label_idx] = 1.0
        replay_data.append({
            "text":   example["text"],
            "labels": label_vec.tolist()
        })
    
    # Combine: corrections + replay
    combined = Dataset.from_list(correction_data + replay_data)
    
    def tokenize(examples):
        return tokenizer(examples["text"], max_length=128, 
                        truncation=True, padding="max_length")
    
    tokenized = combined.map(tokenize, batched=True)
    tokenized.set_format("torch")
    
    # Short fine-tune — just enough to absorb corrections
    args = TrainingArguments(
        output_dir              = "./emotion_model/finetuned",
        num_train_epochs        = 3,
        per_device_train_batch_size = 16,
        learning_rate           = 5e-6,  # small LR — don't overwrite too much
        weight_decay            = 0.01,
        warmup_ratio            = 0.05,
        save_strategy           = "no",
        logging_steps           = 50,
        report_to               = "none"
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model("./emotion_model/final")  # overwrite checkpoint
    tokenizer.save_pretrained("./emotion_model/final")
    
    print(f"Incremental update: {len(corrections)} corrections + "
          f"{len(replay_data)} replay examples")


if __name__ == "__main__":
    corrections = load_corrections()
    incremental_finetune(corrections)
