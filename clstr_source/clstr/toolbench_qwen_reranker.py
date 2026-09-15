from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_RERANK_INSTRUCTION = (
    "Given a ToolBench trajectory state, rank whether the candidate skill is the correct next skill "
    "to execute. Prefer skills that match the current goal, observations, and tool-use history."
)


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 15)].rstrip() + "\n[truncated]"


def format_qwen3_reranker_instruction(*, instruction: str | None, query: str, document: str) -> str:
    task = str(instruction or DEFAULT_RERANK_INSTRUCTION).strip()
    return f"<Instruct>: {task}\n<Query>: {str(query or '').strip()}\n<Document>: {str(document or '').strip()}"


def build_qwen3_reranker_document(skill: dict[str, Any], *, max_skill_chars: int = 900) -> str:
    skill_id = str(skill.get("skill_id") or skill.get("id") or "")
    name = str(skill.get("name") or skill_id)
    description = str(skill.get("description") or skill.get("executor_desc") or "")
    body = str(skill.get("body") or skill.get("skill_md") or skill.get("executor_desc") or "")
    return "\n".join(
        [
            f"skill_id: {skill_id}",
            f"name: {name}",
            f"description: {description}",
            "skill_text:",
            _truncate_text(body, max_skill_chars),
        ]
    )


def rank_candidate_skill_ids_by_scores(*, candidate_skill_ids: list[str], scores: list[float]) -> list[str]:
    pairs = [(idx, str(skill_id), float(score)) for idx, (skill_id, score) in enumerate(zip(candidate_skill_ids, scores))]
    pairs.sort(key=lambda item: (-item[2], item[0]))
    return [skill_id for _, skill_id, _ in pairs]


@dataclass(frozen=True)
class Qwen3RerankerConfig:
    model_name_or_path: str = "models/Qwen3-Reranker-8B"
    torch_dtype: str = "bfloat16"
    local_files_only: bool = True
    trust_remote_code: bool = True
    batch_size: int = 4
    max_length: int = 2048
    max_skill_chars: int = 900
    instruction: str = DEFAULT_RERANK_INSTRUCTION
    score_mode: str = "logit_diff"
    device: str | None = None


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


class Qwen3RerankerScorer:
    def __init__(self, config: Qwen3RerankerConfig) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for Qwen3 reranker scoring")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            padding_side="left",
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            torch_dtype=_resolve_dtype(config.torch_dtype),
        )
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(device).eval()
        self.device = device
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.token_false_id is None or self.token_true_id is None or self.token_false_id == self.tokenizer.unk_token_id or self.token_true_id == self.tokenizer.unk_token_id:
            raise ValueError("Qwen3 reranker tokenizer must contain single-token 'yes' and 'no'")
        self.prefix_tokens = self.tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
            add_special_tokens=False,
        )
        self.suffix_tokens = self.tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )

    def _process_pairs(self, pairs: list[str]) -> dict[str, torch.Tensor]:
        payload_max_length = max(1, int(self.config.max_length) - len(self.prefix_tokens) - len(self.suffix_tokens))
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=payload_max_length,
        )
        for idx, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = self.prefix_tokens + input_ids + self.suffix_tokens
        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=int(self.config.max_length),
        )
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in padded.items()}

    @torch.no_grad()
    def score_pairs(self, *, queries: list[str], documents: list[str]) -> list[float]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")
        if not queries:
            return []
        pairs = [
            format_qwen3_reranker_instruction(
                instruction=self.config.instruction,
                query=query,
                document=document,
            )
            for query, document in zip(queries, documents)
        ]
        scores: list[float] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(pairs), batch_size):
            inputs = self._process_pairs(pairs[start : start + batch_size])
            logits = self.model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            if self.config.score_mode == "yes_probability":
                two_class = torch.stack([false_logits, true_logits], dim=1)
                batch_scores = torch.nn.functional.log_softmax(two_class, dim=1)[:, 1].exp()
            elif self.config.score_mode == "logit_diff":
                batch_scores = true_logits - false_logits
            else:
                raise ValueError(f"unsupported score_mode: {self.config.score_mode}")
            scores.extend(float(item) for item in batch_scores.detach().cpu().tolist())
        return scores
