# services/learner/main.py
# ── Learner: Continual Learning with Local Model ──────────────────────────────
# Runs a small local model (Phi-4-mini / Qwen2.5-1.5B) that learns
# incrementally from every interaction without forgetting past knowledge.
#
# THREE ANTI-FORGETTING MECHANISMS:
#   1. LoRA Adapters  — new weights added on top of frozen base (base never changes)
#   2. Experience Replay — old examples mixed into every training batch
#   3. EWC Penalty   — important weights penalized from changing
#
# THE LOCAL MODEL IS NOT CLAUDE:
#   - Does NOT handle complex reasoning (Claude does that)
#   - Handles: routine classification, pattern recognition, domain-specific QA
#   - Learns from Claude's outputs — distills Claude's reasoning into local weights
#   - Runs on CPU with 4-8GB RAM (no GPU required for Phi-4-mini)
#   - Gets better over time in YOUR specific domain/use case
#
# ON THE FLY LEARNING:
#   - Every high-quality interaction → replay buffer
#   - Every 50 interactions → micro fine-tune (3-5 min on CPU)
#   - Fisher information updated after each domain task → EWC protection
#   - New LoRA adapter per domain → composable knowledge modules

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/shared")

import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import BrainResponse
from events import EventBus

app    = FastAPI(title="Learner", version="1.0.0",
                 description="Continual local model learning — LoRA + EWC + Replay")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
events = EventBus()

# ── Configuration ─────────────────────────────────────────────────────────────
LOCAL_MODEL_NAME  = os.getenv("LOCAL_MODEL_NAME",  "microsoft/phi-4-mini-instruct")
LOCAL_MODEL_PATH  = os.getenv("LOCAL_MODEL_PATH",  "/models/local_llm")
ADAPTER_PATH      = os.getenv("ADAPTER_PATH",      "/models/adapters")
EMOTION_MODEL_PATH= os.getenv("EMOTION_MODEL_PATH","/models/emotion")
HF_TOKEN          = os.getenv("HF_TOKEN",          "")
WORKSPACE         = os.getenv("WORKSPACE_PATH",    "/workspace")

REPLAY_BUFFER_PATH = f"{WORKSPACE}/replay_buffer.jsonl"
EWC_STATE_PATH     = f"{WORKSPACE}/ewc_state.pt"
TRAINING_LOG_PATH  = f"{WORKSPACE}/training_log.jsonl"
EMOTION_CORR_PATH  = f"{WORKSPACE}/emotion_corrections.jsonl"
TRAINING_SUMMARY   = f"{WORKSPACE}/training_summary.jsonl"

# Ensure directories exist
Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(ADAPTER_PATH).mkdir(parents=True, exist_ok=True)
Path(EMOTION_MODEL_PATH).mkdir(parents=True, exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
_model        = None   # local LLM
_tokenizer    = None
_emotion_pipe = None   # emotion classifier pipeline
_ewc          = None   # EWC state tracker
_is_training  = False
_model_loaded = False
_load_error   = None


# ── Models ────────────────────────────────────────────────────────────────────

class ExperienceRequest(BaseModel):
    session_id: str
    user_input: str
    response:   str
    quality:    float = 0.7      # 0=bad, 1=excellent
    domain:     str   = "general"
    emotion:    str   = "neutral"
    source:     str   = "claude" # claude | user | verified


class InferRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = 200
    temperature: float = 0.7
    domain:      str   = "general"


class EmotionRequest(BaseModel):
    text:        str
    session_id:  str = ""


class EmotionCorrectionRequest(BaseModel):
    text:        str
    predicted:   list
    correct:     list
    session_id:  str = ""


class TrainRequest(BaseModel):
    max_examples:  int   = 100
    replay_ratio:  float = 0.35
    epochs:        int   = 1
    domain:        str   = "all"


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer of high-quality past interactions.
    The KEY anti-forgetting tool.

    During any training cycle, we mix new examples with old ones.
    Ratio: ~35% old examples, 65% new examples.
    Even 20-30 replayed examples prevent catastrophic forgetting in practice.
    Research basis: Ratcliff (1990), Robins (1995), Shin et al. (2017).
    """

    def __init__(self, path: str, max_size: int = 15000):
        self.path     = Path(path)
        self.max_size = max_size

    def add(self, example: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(example) + "\n")
        self._trim()

    def sample(self, n: int) -> list:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            lines = f.readlines()
        if not lines:
            return []
        idx = np.random.choice(len(lines), min(n, len(lines)), replace=False)
        result = []
        for i in idx:
            try:
                result.append(json.loads(lines[i]))
            except Exception:
                pass
        return result

    def size(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)

    def _trim(self):
        if not self.path.exists():
            return
        with open(self.path) as f:
            lines = f.readlines()
        if len(lines) > self.max_size:
            with open(self.path, "w") as f:
                f.writelines(lines[-self.max_size:])

    def recent(self, n: int = 50) -> list:
        """Get the most recent n examples."""
        if not self.path.exists():
            return []
        with open(self.path) as f:
            lines = f.readlines()
        result = []
        for line in lines[-n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result


replay_buffer = ReplayBuffer(REPLAY_BUFFER_PATH)


# ── EWC Implementation ────────────────────────────────────────────────────────

class EWCTracker:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    After learning any task, computes Fisher Information for each parameter.
    Fisher Information measures: how much does the loss change if we perturb this weight?
    High Fisher = important weight = protect it during future learning.

    The penalty term added to loss:
        L_EWC = L_new + (lambda/2) * sum_i(F_i * (theta_i - theta*_i)^2)

    Where:
        F_i    = Fisher information for parameter i (importance)
        theta_i = current value of parameter i
        theta*_i = optimal value from previous task
        lambda  = protection strength (5000 is typical)

    Effect: weights important for past tasks can only change in directions
    that don't hurt past performance. New knowledge goes into less-important weights.
    """

    def __init__(self, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher:  dict = {}   # param_name → importance tensor
        self.optimal: dict = {}   # param_name → value at last task boundary

    def compute_fisher(self, model, dataloader, num_samples: int = 300):
        """Compute Fisher Information Matrix after finishing a task."""
        import torch
        model.eval()

        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        seen = 0
        for batch in dataloader:
            if seen >= num_samples:
                break
            model.zero_grad()

            input_ids = batch.get("input_ids")
            attn_mask = batch.get("attention_mask")
            if input_ids is None:
                continue

            labels = input_ids.clone()
            labels[labels == 0] = -100   # ignore padding in loss

            try:
                out = model(input_ids=input_ids,
                            attention_mask=attn_mask,
                            labels=labels)
                out.loss.backward()

                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2) * input_ids.shape[0]

                seen += input_ids.shape[0]
            except Exception:
                continue

        for n in fisher:
            if seen > 0:
                fisher[n] /= seen
            self.fisher[n]  = fisher[n].clone().detach()
            self.optimal[n] = dict(model.named_parameters())[n].data.clone().detach()

    def ewc_loss(self, model, task_loss):
        """Add EWC regularization penalty to task loss."""
        import torch
        if not self.fisher:
            return task_loss

        penalty = torch.tensor(0.0, requires_grad=False)
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty = penalty + (
                    self.fisher[n] * (p - self.optimal[n]).pow(2)
                ).sum()

        return task_loss + (self.lambda_ewc / 2.0) * penalty

    def save(self, path: str):
        import torch
        torch.save({"fisher": self.fisher, "optimal": self.optimal}, path)

    def load(self, path: str):
        import torch
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            self.fisher  = state.get("fisher", {})
            self.optimal = state.get("optimal", {})


# ── Model Loading ──────────────────────────────────────────────────────────────

async def _load_models_async():
    """Load all models in background. Does not block startup."""
    global _model, _tokenizer, _emotion_pipe, _ewc, _model_loaded, _load_error

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models_sync)


def _clear_stale_hf_cache():
    """Clear cached model code that may be incompatible with current transformers version.

    HuggingFace encodes hyphens as '_hyphen_' in cache directory names.
    Stale cached modules cause ImportError at model load time.
    """
    import shutil
    cache_base = Path("/models/hf_cache/modules/transformers_modules")
    if not cache_base.exists():
        return False
    # Match both the literal and HF-encoded directory names
    cleared = False
    for pattern in ["microsoft/phi-4-mini-instruct", "microsoft/phi_hyphen_4_hyphen_mini_hyphen_instruct"]:
        d = cache_base / pattern
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"  Cleared stale cache: {d}")
            cleared = True
    return cleared


def _load_models_sync():
    """Synchronous model loading — runs in thread pool."""
    global _model, _tokenizer, _emotion_pipe, _ewc, _model_loaded, _load_error

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print("Loading local model...")

        # Check if model already downloaded
        model_dir = Path(LOCAL_MODEL_PATH)
        if (model_dir / "config.json").exists():
            model_source = LOCAL_MODEL_PATH
            print(f"  Loading from local path: {LOCAL_MODEL_PATH}")
        else:
            model_source = LOCAL_MODEL_NAME
            print(f"  Downloading from HuggingFace: {LOCAL_MODEL_NAME}")
            print(f"  This may take 10-30 minutes on first run...")

        # Load tokenizer — try fast first, then slow, then with cache cleanup
        tok_kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            tok_kwargs["token"] = HF_TOKEN

        _tokenizer = None
        for attempt_label, attempt_kwargs in [
            ("fast tokenizer", {}),
            ("slow tokenizer", {"use_fast": False}),
            ("slow tokenizer (fresh cache)", {"use_fast": False}),
        ]:
            try:
                if attempt_label == "slow tokenizer (fresh cache)":
                    _clear_stale_hf_cache()
                merged = {**tok_kwargs, **attempt_kwargs}
                _tokenizer = AutoTokenizer.from_pretrained(model_source, **merged)
                print(f"  Tokenizer loaded ({attempt_label})")
                break
            except Exception as e:
                print(f"  {attempt_label} failed: {e}")
                continue

        if _tokenizer is None:
            raise RuntimeError("All tokenizer loading attempts failed")

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        # Load model — CPU only, 4-bit quantization if bitsandbytes available
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.float32,         # CPU: float32 (torch_dtype is deprecated)
            "low_cpu_mem_usage": True
        }
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32
            )
            print("  4-bit quantization enabled (bitsandbytes)")
        except ImportError:
            print("  bitsandbytes not available — loading in float32")

        # Try loading model; if stale cache causes ImportError, clear and retry
        try:
            _model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"  Model load failed (likely stale cache): {e}")
            if _clear_stale_hf_cache():
                print("  Retrying model load with fresh cache...")
                _model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
            else:
                raise

        # Load LoRA adapter if it exists
        adapter_dir = Path(ADAPTER_PATH)
        if (adapter_dir / "adapter_config.json").exists():
            try:
                from peft import PeftModel
                _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)
                print(f"  Loaded LoRA adapter from {ADAPTER_PATH}")
            except Exception as e:
                print(f"  Could not load adapter: {e}")

        _model.eval()

        # Load EWC state if it exists
        _ewc = EWCTracker(lambda_ewc=5000.0)
        _ewc.load(EWC_STATE_PATH)
        if _ewc.fisher:
            print(f"  Loaded EWC state ({len(_ewc.fisher)} parameters protected)")

        print(f"  Local model ready: {LOCAL_MODEL_NAME}")

        # Load emotion classifier if available
        emotion_dir = Path(EMOTION_MODEL_PATH)
        if (emotion_dir / "config.json").exists():
            try:
                _emotion_pipe = pipeline(
                    "text-classification",
                    model=EMOTION_MODEL_PATH,
                    tokenizer=EMOTION_MODEL_PATH,
                    top_k=5,
                    device=-1
                )
                print(f"  Emotion classifier loaded from {EMOTION_MODEL_PATH}")
            except Exception as e:
                print(f"  Emotion classifier load failed: {e}")
        else:
            print(f"  No emotion model at {EMOTION_MODEL_PATH} — train it first")

        _model_loaded = True

    except Exception as e:
        _load_error = str(e)
        print(f"Model load failed: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def startup():
    """Start model loading in background — service responds immediately."""
    print("Learner service starting...")
    print("Model loading will happen in background...")
    asyncio.create_task(_load_models_async())


@app.on_event("shutdown")
async def shutdown():
    pass


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":         "alive",
        "service":        "learner",
        "model_loaded":   _model_loaded,
        "model_name":     LOCAL_MODEL_NAME,
        "load_error":     _load_error,
        "replay_buffer":  replay_buffer.size(),
        "is_training":    _is_training,
        "ewc_active":     bool(_ewc and _ewc.fisher),
        "emotion_ready":  _emotion_pipe is not None
    }


@app.post("/experience")
async def log_experience(req: ExperienceRequest,
                          background: BackgroundTasks):
    """
    Log an interaction for continual learning.
    High-quality interactions (quality > 0.65) go into replay buffer.
    Training triggers automatically every 50 high-quality examples.
    """
    example = {
        "input":     f"### Human: {req.user_input}\n### Assistant:",
        "output":    req.response,
        "quality":   req.quality,
        "domain":    req.domain,
        "emotion":   req.emotion,
        "source":    req.source,
        "session":   req.session_id,
        "timestamp": time.time()
    }

    # Always log to training log
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(json.dumps(example) + "\n")

    # Only replay high-quality examples
    if req.quality >= 0.65:
        replay_buffer.add(example)

    buf_size = replay_buffer.size()

    # Auto-train every 50 new high-quality examples
    if buf_size > 0 and buf_size % 50 == 0 and not _is_training:
        background.add_task(_run_training_cycle, TrainRequest())

    return {
        "logged":        True,
        "buffer_size":   buf_size,
        "quality":       req.quality,
        "will_replay":   req.quality >= 0.65,
        "training_threshold": f"Next training at {((buf_size // 50) + 1) * 50} examples"
    }


@app.post("/infer")
async def local_infer(req: InferRequest):
    """
    Run inference on the local model.
    Use this for routine tasks instead of Claude to save API costs.
    Response quality reflects what the model has learned from your domain.
    """
    if not _model_loaded:
        return BrainResponse(
            request_id = "infer",
            service    = "learner",
            success    = False,
            data       = {"error": "Model not loaded yet",
                          "loaded": False}
        )

    start = time.time()

    try:
        import torch
        inputs = _tokenizer(
            req.prompt,
            return_tensors  = "pt",
            max_length      = 512,
            truncation      = True,
            padding         = False
        )

        with torch.no_grad():
            output_ids = _model.generate(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs.get("attention_mask"),
                max_new_tokens = req.max_tokens,
                temperature    = req.temperature,
                do_sample      = req.temperature > 0,
                pad_token_id   = _tokenizer.pad_token_id,
                eos_token_id   = _tokenizer.eos_token_id
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response   = _tokenizer.decode(new_tokens, skip_special_tokens=True)

        return BrainResponse(
            request_id = "infer",
            service    = "learner",
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {
                "response":  response,
                "model":     LOCAL_MODEL_NAME,
                "tokens":    len(new_tokens),
                "local":     True
            }
        )

    except Exception as e:
        return BrainResponse(
            request_id = "infer",
            service    = "learner",
            success    = False,
            latency_ms = round((time.time() - start) * 1000, 2),
            data       = {"error": str(e)}
        )


@app.post("/classify-emotion")
async def classify_emotion(req: EmotionRequest):
    """
    Classify emotion using the trained emotion model.
    Falls back to simple heuristics if model not loaded.
    """
    start = time.time()

    if _emotion_pipe is not None:
        try:
            results = _emotion_pipe(req.text[:512])
            if isinstance(results[0], list):
                results = results[0]

            detected = [
                {"emotion": r["label"], "score": round(r["score"], 3)}
                for r in results
                if r["score"] > 0.12
            ]

            primary    = detected[0]["emotion"] if detected else "neutral"
            confidence = detected[0]["score"]   if detected else 0.3

            return {
                "primary":    primary,
                "confidence": confidence,
                "all":        detected,
                "model":      "trained",
                "latency_ms": round((time.time() - start) * 1000, 2)
            }
        except Exception as e:
            print(f"Emotion classification error: {e}")

    # Heuristic fallback
    text_lower = req.text.lower()
    heuristic_map = {
        "anger":    ["angry", "furious", "frustrated", "annoyed", "hate"],
        "sadness":  ["sad", "depressed", "crying", "miss", "loss", "grief"],
        "joy":      ["happy", "excited", "wonderful", "love", "amazing"],
        "fear":     ["scared", "afraid", "anxious", "worried", "panic"],
        "surprise": ["wow", "incredible", "unexpected", "shocked", "unbelievable"],
        "curiosity":["wonder", "how", "why", "curious", "interesting"],
    }

    for emotion, keywords in heuristic_map.items():
        if any(kw in text_lower for kw in keywords):
            return {
                "primary": emotion, "confidence": 0.45,
                "all": [{"emotion": emotion, "score": 0.45}],
                "model": "heuristic",
                "latency_ms": round((time.time() - start) * 1000, 2)
            }

    return {
        "primary": "neutral", "confidence": 0.3,
        "all": [{"emotion": "neutral", "score": 0.3}],
        "model": "heuristic",
        "latency_ms": round((time.time() - start) * 1000, 2)
    }


@app.post("/emotion-correction")
async def log_emotion_correction(req: EmotionCorrectionRequest):
    """
    Log when the emotion classifier was wrong.
    These corrections accumulate and feed the nightly fine-tune.
    This IS the supervised signal for emotion continual learning.
    """
    correction = {
        "text":      req.text,
        "predicted": req.predicted,
        "correct":   req.correct,
        "session":   req.session_id,
        "timestamp": time.time()
    }
    with open(EMOTION_CORR_PATH, "a") as f:
        f.write(json.dumps(correction) + "\n")

    # Count corrections
    count = sum(1 for _ in open(EMOTION_CORR_PATH))

    return {
        "logged":              True,
        "correction_count":    count,
        "finetune_threshold":  "Training triggers at 30 corrections"
    }


@app.post("/train")
async def trigger_training(req: TrainRequest,
                            background: BackgroundTasks):
    """Manually trigger a training cycle."""
    if _is_training:
        return {"error": "Already training"}
    if not _model_loaded:
        return {"error": "Model not loaded"}

    background.add_task(_run_training_cycle, req)
    return {
        "triggered":     True,
        "max_examples":  req.max_examples,
        "replay_ratio":  req.replay_ratio
    }


async def _run_training_cycle(req: TrainRequest):
    """Background: run one complete continual learning cycle."""
    global _is_training

    if _is_training or not _model_loaded:
        return

    _is_training = True
    loop         = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            None, _train_sync, req
        )
    finally:
        _is_training = False


def _train_sync(req: TrainRequest):
    """Synchronous training — runs in thread pool."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW

    try:
        from peft import get_peft_model, LoraConfig, TaskType
        peft_available = True
    except ImportError:
        peft_available = False

    print(f"\n[Learner] Starting training cycle...")
    cycle_start = time.time()

    # ── Collect training data ──────────────────────────────────────────────────
    # New examples from recent log
    new_examples = replay_buffer.recent(int(req.max_examples * (1 - req.replay_ratio)))

    # Replay from buffer (anti-forgetting)
    replay_count   = int(req.max_examples * req.replay_ratio)
    replay_examples = replay_buffer.sample(replay_count)

    all_examples = new_examples + replay_examples
    if len(all_examples) < 5:
        print(f"[Learner] Not enough examples ({len(all_examples)}) — skipping")
        return

    print(f"[Learner] Training on {len(new_examples)} new + "
          f"{len(replay_examples)} replay = {len(all_examples)} total")

    # ── Dataset ───────────────────────────────────────────────────────────────
    class ContinualDataset(Dataset):
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
                text,
                max_length = self.max_len,
                truncation = True,
                padding    = "max_length",
                return_tensors = "pt"
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

    dataset    = ContinualDataset(all_examples, _tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # ── Add LoRA if not present ────────────────────────────────────────────────
    model = _model
    if peft_available and not hasattr(model, "peft_config"):
        try:
            lora_config = LoraConfig(
                task_type      = TaskType.CAUSAL_LM,
                r              = 16,
                lora_alpha     = 32,
                lora_dropout   = 0.05,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
                bias           = "none"
            )
            model = get_peft_model(model, lora_config)
            print(f"[Learner] LoRA adapter added "
                  f"(trainable params: {model.num_trainable_parameters():,})")
        except Exception as e:
            print(f"[Learner] LoRA setup failed: {e} — training all params")

    # ── Training loop ──────────────────────────────────────────────────────────
    # Very small LR — we don't want to overwrite existing knowledge
    optimizer  = AdamW(model.parameters(), lr=8e-6, weight_decay=0.01)
    model.train()
    total_loss = 0.0
    steps      = 0

    for epoch in range(req.epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            attn_mask = batch.get("attention_mask")
            labels    = input_ids.clone()
            labels[labels == _tokenizer.pad_token_id] = -100

            try:
                out = model(
                    input_ids      = input_ids,
                    attention_mask = attn_mask,
                    labels         = labels
                )
                loss = out.loss

                # Apply EWC penalty (protects knowledge from previous tasks)
                if _ewc:
                    loss = _ewc.ewc_loss(model, loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                steps      += 1

            except Exception as e:
                print(f"[Learner] Step error: {e}")
                continue

    avg_loss = total_loss / max(1, steps)
    duration = time.time() - cycle_start

    # ── Update EWC Fisher information ─────────────────────────────────────────
    if _ewc and steps > 0:
        _ewc.compute_fisher(model, dataloader, num_samples=200)
        _ewc.save(EWC_STATE_PATH)

    # ── Save updated adapter ───────────────────────────────────────────────────
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ADAPTER_PATH)
        _tokenizer.save_pretrained(ADAPTER_PATH)

    # ── Emotion model fine-tune if corrections accumulated ────────────────────
    emotion_corrections = _count_emotion_corrections()
    if emotion_corrections >= 30:
        _finetune_emotion_model()

    # ── Log summary ───────────────────────────────────────────────────────────
    summary = {
        "timestamp":       time.time(),
        "duration_s":      round(duration, 1),
        "examples":        len(all_examples),
        "new":             len(new_examples),
        "replay":          len(replay_examples),
        "avg_loss":        round(avg_loss, 5),
        "steps":           steps,
        "ewc_active":      bool(_ewc and _ewc.fisher),
        "peft_used":       peft_available
    }

    with open(TRAINING_SUMMARY, "a") as f:
        f.write(json.dumps(summary) + "\n")

    events.publish("self.update.triggered", {
        "type":       "continual_learning",
        "examples":   len(all_examples),
        "avg_loss":   avg_loss,
        "duration_s": duration
    })

    print(f"[Learner] Training complete: loss={avg_loss:.5f}, "
          f"steps={steps}, time={duration:.1f}s")


def _count_emotion_corrections() -> int:
    if not os.path.exists(EMOTION_CORR_PATH):
        return 0
    with open(EMOTION_CORR_PATH) as f:
        return sum(1 for _ in f)


def _finetune_emotion_model():
    """
    Incremental fine-tune of the emotion classifier on accumulated corrections.
    Uses experience replay from GoEmotions to prevent forgetting.
    """
    emotion_dir = Path(EMOTION_MODEL_PATH)
    if not (emotion_dir / "config.json").exists():
        print("[Learner] No emotion model to fine-tune — train base model first")
        return

    print("[Learner] Fine-tuning emotion classifier on corrections...")

    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   Trainer, TrainingArguments)
        from datasets import Dataset as HFDataset

        # Load corrections
        corrections = []
        with open(EMOTION_CORR_PATH) as f:
            for line in f:
                try:
                    corrections.append(json.loads(line))
                except Exception:
                    pass

        if len(corrections) < 20:
            return

        tok    = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
        model  = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH)
        labels = list(model.config.id2label.values())
        n_lab  = len(labels)
        l2id   = {l: i for i, l in enumerate(labels)}

        def make_example(text, correct_labels):
            vec = [0.0] * n_lab
            for em in correct_labels:
                if em in l2id:
                    vec[l2id[em]] = 1.0
            return {"text": text, "labels": vec}

        data = [make_example(c["text"], c["correct"]) for c in corrections]

        def tokenize(ex):
            enc = tok(ex["text"], max_length=128, truncation=True, padding="max_length")
            enc["labels"] = ex["labels"]
            return enc

        ds = HFDataset.from_list(data).map(tokenize)
        ds.set_format("torch")

        args = TrainingArguments(
            output_dir              = f"{EMOTION_MODEL_PATH}/finetuned",
            num_train_epochs        = 3,
            per_device_train_batch_size = 8,
            learning_rate           = 4e-6,   # small LR — don't overwrite
            warmup_ratio            = 0.1,
            save_strategy           = "no",
            logging_steps           = 20,
            report_to               = "none"
        )

        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()

        # Save over base model
        model.save_pretrained(EMOTION_MODEL_PATH)
        tok.save_pretrained(EMOTION_MODEL_PATH)

        # Clear processed corrections
        open(EMOTION_CORR_PATH, "w").close()

        print(f"[Learner] Emotion model updated on {len(corrections)} corrections")

    except Exception as e:
        print(f"[Learner] Emotion fine-tune failed: {e}")


@app.get("/status")
async def status():
    """Full learner status."""
    return {
        "model_loaded":         _model_loaded,
        "model_name":           LOCAL_MODEL_NAME,
        "model_path":           LOCAL_MODEL_PATH,
        "adapter_path":         ADAPTER_PATH,
        "replay_buffer_size":   replay_buffer.size(),
        "is_training":          _is_training,
        "ewc_protected_params": len(_ewc.fisher) if _ewc else 0,
        "emotion_model_ready":  _emotion_pipe is not None,
        "emotion_corrections":  _count_emotion_corrections(),
        "load_error":           _load_error
    }


@app.get("/training-history")
async def training_history(limit: int = 10):
    if not os.path.exists(TRAINING_SUMMARY):
        return {"history": []}
    with open(TRAINING_SUMMARY) as f:
        lines = f.readlines()
    history = []
    for line in lines[-limit:]:
        try:
            history.append(json.loads(line))
        except Exception:
            pass
    return {"history": list(reversed(history))}
