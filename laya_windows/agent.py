"""PyTorch inference for Windows/Linux (CPU or CUDA). Replaces the Core ML runtime."""

import json
from pathlib import Path

import numpy as np
import torch

from .common import read_temperatures
from .prompt import PromptMixin
from .result import ResultMixin
from .tokenizer import Tokenizer
from .torch_model import load_model

# Original upstream checkpoints, pinned to the revisions laya-coreml validated against.
REVISIONS = {
    "convaiinnovations/laya": "c5d78730f3493e4fe16d61507ef4b78eef7318cf",
    "convaiinnovations/laya-multilingual": "052592a15d198d9ad47da779604259b10b47b7aa",
    "convaiinnovations/laya-typed-decisions": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
}
DEFAULT_MODEL = "convaiinnovations/laya-multilingual"


def resolve_checkpoint(model, *, revision=None, local_files_only=False):
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    from huggingface_hub import snapshot_download

    repo_id = str(model) if "/" in str(model) else "convaiinnovations/" + str(model)
    return Path(
        snapshot_download(
            repo_id,
            revision=revision or REVISIONS.get(repo_id),
            local_files_only=local_files_only,
            allow_patterns=[
                "model.safetensors",
                "encoder/config.json",
                "rl_agent_config.json",
                "tokenizer/*",
            ],
        )
    )


class Agent(PromptMixin, ResultMixin):
    def __init__(self, model_dir=DEFAULT_MODEL, *, device=None, revision=None,
                 local_files_only=False, batch_size=8):
        self.model_dir = resolve_checkpoint(
            model_dir, revision=revision, local_files_only=local_files_only
        )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        (
            self.temperature,
            self.temperature_by_options,
            self.temperature_raw,
            self.temperature_by_options_raw,
        ) = read_temperatures(self.cfg)
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        max_len = self.cfg.get("max_len", 512)
        self.batch_size = batch_size
        self.pad_to_multiple = 16
        # Dynamic shapes: pad each batch to the next multiple of 16 tokens.
        self.shape = {"batch_size": batch_size, "max_length": max_len, "max_options": 64,
                      "flexible": True, "min_length": 16}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_model(self.model_dir, max_len, "sdpa").to(self.device)

    @torch.inference_mode()
    def forward(self, batch):
        tensors = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        logits, action = self.model(**tensors)
        return logits.float().cpu().numpy().astype(np.float32), action.float().cpu().numpy()


def load(model_dir=DEFAULT_MODEL, **kwargs):
    return Agent(model_dir, **kwargs)
