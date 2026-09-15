from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


def _resolve_dtype(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported torch_dtype: {value}")
    return mapping[value]


def bge_reranker_scores_from_logits(logits: torch.Tensor) -> list[float]:
    if logits.ndim != 2:
        raise ValueError(f"expected sequence-classification logits with shape [batch, labels], got {tuple(logits.shape)}")
    if logits.size(-1) == 1:
        scores = logits[:, 0]
    elif logits.size(-1) == 2:
        scores = logits[:, 1] - logits[:, 0]
    else:
        raise ValueError(f"unsupported BGE reranker label count: {logits.size(-1)}")
    return [float(item) for item in scores.detach().float().cpu().tolist()]


def bge_reranker_score_tensor(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"expected sequence-classification logits with shape [batch, labels], got {tuple(logits.shape)}")
    if logits.size(-1) == 1:
        return logits[:, 0]
    if logits.size(-1) == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(f"unsupported BGE reranker label count: {logits.size(-1)}")


@dataclass(frozen=True)
class BGERerankerConfig:
    model_name_or_path: str = "models/BAAI/bge-reranker-v2-m3"
    torch_dtype: str = "bfloat16"
    local_files_only: bool = True
    trust_remote_code: bool = True
    batch_size: int = 16
    max_length: int = 2048
    device: str | None = None


class BGERerankerScorer:
    def __init__(self, config: BGERerankerConfig) -> None:
        if AutoModelForSequenceClassification is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for BGE reranker scoring")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            torch_dtype=_resolve_dtype(config.torch_dtype),
        )
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score_pairs(self, *, queries: list[str], documents: list[str]) -> list[float]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")
        if not queries:
            return []
        scores: list[float] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(queries), batch_size):
            batch_queries = [str(item or "") for item in queries[start : start + batch_size]]
            batch_documents = [str(item or "") for item in documents[start : start + batch_size]]
            tokens = self.tokenizer(
                batch_queries,
                batch_documents,
                padding=True,
                truncation=True,
                max_length=int(self.config.max_length),
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            logits = self.model(**tokens).logits
            scores.extend(bge_reranker_scores_from_logits(logits))
        return scores
