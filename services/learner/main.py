# ── Learner: Continual Learning Service ────────────────────────────────────────
# Manages a local model that trains incrementally from interactions.
# Uses three anti-forgetting mechanisms simultaneously:
#   1. Experience Replay — mix new with old examples
#   2. EWC (Elastic Weight Consolidation) — protect important weights
#   3. LoRA adapters — add new capability without touching base weights

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/shared")

import torch
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="Learner", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()

MODEL_PATH     = os.getenv("LOCAL_MODEL_PATH", "/models/local_llm")
ADAPTER_PATH   = os.getenv("ADAPTER_PATH",     "/models/adapters")
REPLAY_BUFFER  = "/workspace/replay_buffer.jsonl"
EWC_STATE      = "/workspace/ewc_state.pt"
TRAINING_LOG   = "/workspace/training_log.jsonl"

_model     = None
_tokenizer = None
_ewc_state = {}    # Fisher information + optimal weights
_is_training = False


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer of past experiences.
    During new training, mix in old examples to prevent forgetting.
    
    The key insight from continual learning research:
    Replaying just 5-10% of past data is enough to prevent catastrophic forgetting
    in many settings (Ratcliff 1990, McClelland 1995, Robins 1995).
    """
    
    def __init__(self, path: str, max_size: int = 10000):
        self.path     = Path(path)
        self.max_size = max_size
    
    def add(self, example: dict):
        """Add an example to the replay buffer."""
        with open(self.path, "a") as f:
            f.write(json.dumps(example) + "\n")
        self._enforce_limit()
    
    def sample(self, n: int) -> list:
        """Sample n random examples from the buffer."""
        if not self.path.exists():
            return []
        
        with open(self.path) as f:
            lines = f.readlines()
        
        if not lines:
            return []
        
        indices = np.random.choice(len(lines), min(n, len(lines)), replace=False)
        return [json.loads(lines[i]) for i in indices]
    
    def size(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)
    
    def _enforce_limit(self):
        """Keep buffer at max_size by removing oldest examples."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            lines = f.readlines()
        if len(lines) > self.max_size:
            with open(self.path, "w") as f:
                f.writelines(lines[-self.max_size:])


replay_buffer = ReplayBuffer(REPLAY_BUFFER)


# ── EWC Implementation ────────────────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).
    
    After learning task A, measures how important each weight is
    (via Fisher Information Matrix). When learning task B, penalizes
    changes to weights that were important for task A.
    
    The intuition: some weights are crucial for past knowledge.
    Protect them. Let others change freely.
    """
    
    def __init__(self, model, lambda_ewc: float = 5000.0):
        self.model      = model
        self.lambda_ewc = lambda_ewc   # higher = stronger protection
        self.fisher     = {}           # parameter name → importance
        self.optimal    = {}           # parameter name → value at last task
    
    def compute_fisher(self, dataloader, num_samples: int = 500):
        """
        Compute Fisher Information for current parameters.
        Must be called AFTER learning a task, BEFORE starting the next.
        """
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            self.model.zero_grad()
            
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = input_ids.clone()   # language model: predict next token
            
            output = self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )
            
            output.loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2) * input_ids.shape[0]
            
            samples_seen += input_ids.shape[0]
        
        # Normalize
        for n in fisher:
            fisher[n] /= samples_seen
            self.fisher[n] = fisher[n].clone().detach()
        
        # Save optimal weights (current parameters)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optimal[n] = p.data.clone().detach()
    
    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """
        Add EWC penalty to loss.
        Penalizes deviation from important weights.
        """
        if not self.fisher:
            return current_loss
        
        penalty = torch.tensor(0.0, requires_grad=True)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                penalty = penalty + (
                    self.fisher[n] * (p - self.optimal[n]).pow(2)
                ).sum()
        
        return current_loss + (self.lambda_ewc / 2) * penalty
    
    def save(self, path: str):
        torch.save({
            "fisher":  self.fisher,
            "optimal": self.optimal
        }, path)
    
    def load(self, path: str):
        if os.path.exists(path):
            state    = torch.load(path, map_location="cpu")
            self.fisher  = state["fisher"]
            self.optimal = state["optimal"]


# ── LoRA Setup ────────────────────────────────────────────────────────────────

def add_lora_adapter(model, rank: int = 16, alpha: int = 32):
    """
    Add a LoRA adapter to the model.
    LoRA freezes the base model and trains tiny low-rank matrices.
    
    This means:
    - Base model knowledge is perfectly preserved (frozen)
    - New knowledge goes into the adapters (small, fast to train)
    - Adapters can be saved/loaded independently
    - Multiple adapters can be composed for multiple domains
    
    For a 7B model with rank=16:
    - Base model: ~14GB (frozen, never changes)
    - LoRA adapter: ~100MB (trains in minutes on CPU)
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        
        config = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = rank,
            lora_alpha     = alpha,
            lora_dropout   = 0.05,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            bias           = "none"
        )
        return get_peft_model(model, config)
    except ImportError:
        print("peft not installed — LoRA not available")
        return model


# ── Training Loop ─────────────────────────────────────────────────────────────

class ExperienceRequest(BaseModel):
    session_id:   str
    user_input:   str
    response:     str
    quality:      float = 0.7    # 0.0 = bad response, 1.0 = excellent
    domain:       str   = "general"
    emotion:      str   = "neutral"


class TrainRequest(BaseModel):
    max_examples: int   = 100
    epochs:       int   = 1
    replay_ratio: float = 0.3    # fraction of batch from replay buffer


@app.get("/health")
async def health():
    return {
        "status":          "alive",
        "service":         "learner",
        "model_loaded":    _model is not None,
        "replay_buffer":   replay_buffer.size(),
        "is_training":     _is_training,
        "ewc_active":      bool(_ewc_state)
    }


@app.post("/experience")
async def log_experience(req: ExperienceRequest,
                          background: BackgroundTasks):
    """
    Log an interaction as a training example.
    High-quality interactions go into replay buffer immediately.
    Low-quality ones are still logged for analysis.
    """
    example = {
        "input":     f"User: {req.user_input}\nAssistant:",
        "output":    req.response,
        "quality":   req.quality,
        "domain":    req.domain,
        "emotion":   req.emotion,
        "session":   req.session_id,
        "timestamp": time.time()
    }
    
    # Only add high-quality examples to replay buffer
    if req.quality > 0.65:
        replay_buffer.add(example)
    
    # Log everything for analysis
    with open(TRAINING_LOG, "a") as f:
        f.write(json.dumps(example) + "\n")
    
    # Trigger training if buffer has grown enough
    if replay_buffer.size() % 50 == 0 and replay_buffer.size() > 0:
        background.add_task(maybe_train)
    
    return {
        "logged":        True,
        "buffer_size":   replay_buffer.size(),
        "quality":       req.quality,
        "will_replay":   req.quality > 0.65
    }


async def maybe_train():
    """Trigger incremental training if conditions are right."""
    global _is_training
    
    if _is_training:
        return
    if _model is None:
        return
    if replay_buffer.size() < 20:
        return
    
    _is_training = True
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, run_training_cycle
        )
    finally:
        _is_training = False


def run_training_cycle(max_examples: int = 50, 
                        replay_ratio: float = 0.3):
    """
    One cycle of continual learning.
    Mixes new examples with replay. Uses EWC penalty.
    """
    if _model is None:
        return
    
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    
    # Sample new examples from log
    new_examples  = []
    replay_examples = replay_buffer.sample(int(max_examples * replay_ratio))
    
    # Read recent training log
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG) as f:
            lines = f.readlines()[-max_examples:]
        new_examples = [json.loads(l) for l in lines
                        if json.loads(l)["quality"] > 0.65]
    
    all_examples = new_examples + replay_examples
    if not all_examples:
        return
    
    # Simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, examples, tokenizer, max_len=512):
            self.examples  = examples
            self.tokenizer = tokenizer
            self.max_len   = max_len
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            ex   = self.examples[idx]
            text = ex["input"] + ex["output"] + self.tokenizer.eos_token
            enc  = self.tokenizer(
                text, max_length=self.max_len,
                truncation=True, padding="max_length",
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}
    
    dataset    = SimpleDataset(all_examples, _tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize EWC with saved state
    ewc = EWC(_model, lambda_ewc=5000.0)
    ewc.load(EWC_STATE)
    
    optimizer = AdamW(_model.parameters(), lr=1e-5, weight_decay=0.01)
    
    _model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = input_ids.clone()
        labels[labels == _tokenizer.pad_token_id] = -100
        
        outputs = _model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )
        
        # EWC-regularized loss: task loss + penalty for forgetting
        loss = ewc.ewc_loss(outputs.loss)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Update Fisher information for next cycle
    ewc.compute_fisher(dataloader, num_samples=200)
    ewc.save(EWC_STATE)
    
    # Save updated adapter
    if hasattr(_model, "save_pretrained"):
        _model.save_pretrained(ADAPTER_PATH)
    
    avg_loss = total_loss / max(1, len(dataloader))
    
    with open(TRAINING_LOG + ".summary", "a") as f:
        f.write(json.dumps({
            "timestamp":    time.time(),
            "examples":     len(all_examples),
            "new":          len(new_examples),
            "replay":       len(replay_examples),
            "avg_loss":     avg_loss
        }) + "\n")
    
    events.publish("self.update.triggered", {
        "type":    "continual_learning",
        "examples": len(all_examples),
        "loss":    avg_loss
    })
    
    print(f"Training cycle: {len(all_examples)} examples, "
          f"avg_loss={avg_loss:.4f}")
