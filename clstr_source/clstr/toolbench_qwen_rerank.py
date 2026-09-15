from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


_LABEL_RE = re.compile(r"\bC\d{3}\b", flags=re.IGNORECASE)


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 15)].rstrip() + "\n[truncated]"


def parse_qwen_ranked_labels(response: str, allowed_labels: list[str]) -> list[str]:
    allowed = [str(item).upper() for item in allowed_labels]
    allowed_set = set(allowed)
    raw = str(response or "").strip()
    candidates: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            for match in _LABEL_RE.findall(value.upper()):
                candidates.append(match)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key in ("ranking", "ranked", "answer", "candidates", "skills"):
                if key in value:
                    collect(value[key])

    try:
        collect(json.loads(raw))
    except Exception:
        collect(raw)
    output: list[str] = []
    seen: set[str] = set()
    for label in candidates:
        if label in allowed_set and label not in seen:
            output.append(label)
            seen.add(label)
    return output


def build_qwen_skill_rerank_prompt(
    *,
    state_text: str,
    candidates: list[dict[str, Any]],
    max_skill_chars: int = 900,
) -> str:
    candidate_blocks: list[str] = []
    for item in candidates:
        body = str(item.get("body") or item.get("skill_md") or item.get("description") or "")
        candidate_blocks.append(
            "\n".join(
                [
                    f"[{item['label']}]",
                    f"skill_id: {item.get('skill_id', '')}",
                    f"name: {item.get('name', '')}",
                    f"description: {item.get('description', '')}",
                    "skill_text:",
                    _truncate_text(body, max_skill_chars),
                ]
            )
        )
    labels = [str(item["label"]) for item in candidates]
    return "\n\n".join(
        [
            "You are ranking candidate skills for a ToolBench agent.",
            "Task: choose the next skill that should be used at the current trajectory step.",
            "Use the goal, current state, previous tool history, and each candidate skill's text.",
            "Return JSON only, with this exact schema:",
            '{"ranking": ["C001", "C002", "..."]}',
            "The ranking must list candidate labels from best to worst. Do not include labels outside the candidate set.",
            "",
            "[Current trajectory state]",
            str(state_text or "").strip(),
            "",
            f"[Candidate labels]\n{', '.join(labels)}",
            "",
            "[Candidate skills]",
            "\n\n".join(candidate_blocks),
            "",
            "Return JSON only.",
        ]
    )


def ranking_metrics_from_ranked_skill_ids(
    *,
    ranked_skill_ids: list[str],
    candidate_skill_ids: list[str],
    positive_skill_id: str,
    source_row_count: int = 1,
) -> dict[str, float]:
    generated_count = len(ranked_skill_ids)
    final_ranked: list[str] = []
    seen: set[str] = set()
    candidate_set = set(candidate_skill_ids)
    for skill_id in ranked_skill_ids:
        skill_id = str(skill_id)
        if skill_id in candidate_set and skill_id not in seen:
            final_ranked.append(skill_id)
            seen.add(skill_id)
    for skill_id in candidate_skill_ids:
        skill_id = str(skill_id)
        if skill_id not in seen:
            final_ranked.append(skill_id)
            seen.add(skill_id)
    positive = str(positive_skill_id)
    if positive in final_ranked:
        rank = final_ranked.index(positive) + 1
        recall1 = 1.0 if rank <= 1 else 0.0
        recall5 = 1.0 if rank <= 5 else 0.0
        mrr = 1.0 / rank
    else:
        rank = 0
        recall1 = 0.0
        recall5 = 0.0
        mrr = 0.0
    return {
        "recall@1": recall1,
        "recall@5": recall5,
        "mrr": mrr,
        "rank": float(rank),
        "generated_rank_count": float(generated_count),
        "candidate_count": float(len(candidate_skill_ids)),
    }


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0,
            "rank": 0.0,
            "generated_rank_count": 0.0,
            "candidate_count": 0.0,
        }
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def strict_metrics(metrics: dict[str, float], *, retained_rows: int, source_rows: int) -> dict[str, float]:
    source_rows = max(0, int(source_rows))
    retained_rows = max(0, int(retained_rows))
    scale = retained_rows / source_rows if source_rows > 0 else 0.0
    return {
        "strict_recall@1": float(metrics.get("recall@1") or 0.0) * scale,
        "strict_recall@5": float(metrics.get("recall@5") or 0.0) * scale,
        "strict_mrr": float(metrics.get("mrr") or 0.0) * scale,
        "retained_row_fraction": scale,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
    }


@dataclass(frozen=True)
class QwenSkillRerankConfig:
    model_name_or_path: str = "models/Qwen3-14B"
    torch_dtype: str = "bfloat16"
    local_files_only: bool = True
    trust_remote_code: bool = True
    max_new_tokens: int = 192
    max_skill_chars: int = 900
    enable_thinking: bool = False
    batch_size: int = 1
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


class QwenSkillReranker:
    def __init__(self, config: QwenSkillRerankConfig) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for Qwen skill reranking")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            torch_dtype=_resolve_dtype(config.torch_dtype),
        )
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(device).eval()
        self.device = device

    def _chat_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a precise skill reranker. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.config.enable_thinking),
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def rerank(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        chat_prompts = [self._chat_prompt(prompt) for prompt in prompts]
        outputs: list[str] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(chat_prompts), batch_size):
            batch = chat_prompts[start : start + batch_size]
            tokenized = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            tokenized = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()}
            input_len = tokenized["input_ids"].shape[-1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **tokenized,
                    max_new_tokens=int(self.config.max_new_tokens),
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            generated = generated[:, input_len:]
            outputs.extend(str(item).strip() for item in self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return outputs
