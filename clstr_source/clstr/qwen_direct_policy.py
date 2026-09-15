from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - import availability is tested through callers
    AutoModelForCausalLM = None
    AutoTokenizer = None


_ACTION_PREFIX_RE = re.compile(r"^\s*(?:action|answer|chosen action)\s*:\s*", flags=re.IGNORECASE)
_INDEX_RE = re.compile(r"^\s*(?:#|index\s*:|option\s*:)?\s*(\d{1,3})\s*[.)]?\s*$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class QwenDirectPolicyConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    cache_dir: str | None = None
    local_files_only: bool = False
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_new_tokens: int = 16
    fallback_strategy: str = "first_admissible"
    prompt_template_id: str = "alfworld_admissible_exact_v1"
    device: str | None = None
    likelihood_batch_size: int = 4
    use_chat_template: bool = True
    enable_thinking: bool = False


@dataclass(frozen=True)
class ParsedAction:
    action: str
    status: str
    fallback_used: bool
    invalid_response: str | None = None


def _normalise_action(text: str) -> str:
    text = _ACTION_PREFIX_RE.sub("", str(text or "").strip())
    text = text.strip().strip("`'\"")
    text = re.sub(r"[.。；;]+$", "", text).strip()
    return " ".join(text.lower().split())


def _fallback_action(candidates: list[str], strategy: str) -> str:
    if not candidates:
        return ""
    if strategy == "first_admissible":
        return candidates[0]
    if strategy == "look_if_available_else_first":
        for candidate in candidates:
            if str(candidate).strip().lower() == "look":
                return candidate
        return candidates[0]
    raise ValueError(f"unsupported Qwen direct fallback_strategy: {strategy}")


def parse_admissible_action_response(
    response: str,
    admissible_actions: list[str],
    fallback_strategy: str = "first_admissible",
) -> ParsedAction:
    candidates = [str(item) for item in admissible_actions]
    raw = str(response or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()] or [raw]
    candidate_set = set(candidates)
    normalized_to_action = {_normalise_action(action): action for action in candidates}

    def parse_line(line: str) -> ParsedAction | None:
        stripped = _ACTION_PREFIX_RE.sub("", line).strip().strip("`'\"").strip()
        stripped = re.sub(r"[.。；;]+$", "", stripped).strip()
        if stripped in candidate_set:
            return ParsedAction(action=stripped, status="exact_match", fallback_used=False)
        normalized = _normalise_action(line)
        if normalized in normalized_to_action:
            return ParsedAction(action=normalized_to_action[normalized], status="normalized_match", fallback_used=False)
        numbered_action = re.match(r"^\s*(\d{1,3})\s*[.)]\s+(.+?)\s*$", stripped)
        if numbered_action:
            action_text = numbered_action.group(2).strip().strip("`'\"").strip()
            action_text = re.sub(r"[.。；;]+$", "", action_text).strip()
            if action_text in candidate_set:
                return ParsedAction(action=action_text, status="numbered_action_match", fallback_used=False)
            normalized_action = _normalise_action(action_text)
            if normalized_action in normalized_to_action:
                return ParsedAction(
                    action=normalized_to_action[normalized_action],
                    status="numbered_action_match",
                    fallback_used=False,
                )
        index_match = _INDEX_RE.search(stripped)
        if index_match:
            idx = int(index_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return ParsedAction(action=candidates[idx], status="index_match", fallback_used=False)
        return None

    tagged_lines = [line for line in lines if _ACTION_PREFIX_RE.match(line)]
    for line in tagged_lines + [line for line in lines if line not in tagged_lines]:
        parsed = parse_line(line)
        if parsed is not None:
            return parsed

    fallback = _fallback_action(candidates, fallback_strategy)
    return ParsedAction(
        action=fallback,
        status="fallback",
        fallback_used=True,
        invalid_response=raw,
    )


def build_qwen_direct_prompt(
    state_text: str,
    admissible_actions: list[str],
    prompt_template_id: str = "alfworld_admissible_exact_v1",
    tokenizer: Any | None = None,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> str:
    if prompt_template_id != "alfworld_admissible_exact_v1":
        raise ValueError(f"unsupported prompt_template_id: {prompt_template_id}")
    action_lines = "\n".join(f"{idx}. {action}" for idx, action in enumerate(admissible_actions, start=1))
    user_content = "\n".join(
        [
            "You are choosing the next ALFWorld text action.",
            "You must choose exactly one command from the admissible commands.",
            "Return exactly one line in this format: ACTION: <exact command>",
            "",
            "[State]",
            str(state_text),
            "",
            "AVAILABLE ACTIONS:",
            action_lines,
            "",
            "ACTION:",
        ]
    )
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": "You are an ALFWorld agent. Choose one valid text action exactly from AVAILABLE ACTIONS.",
            },
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return user_content


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


def _move_tokenized(tokenized: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()}


def _runtime_device(config_device: str | None) -> torch.device:
    if config_device:
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QwenDirectAdmissibleActionScorer:
    """Direct Qwen baseline constrained to ALFWorld admissible commands.

    This is intentionally not a CLSTR policy. It returns candidate scores only so it can
    reuse the closed-loop ALFWorld harness while metrics identify it as a reference
    baseline.
    """

    def __init__(
        self,
        config: QwenDirectPolicyConfig | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        backend: Any | None = None,
    ) -> None:
        self.config = config or QwenDirectPolicyConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.last_metadata: list[dict[str, Any]] = []
        if self.backend is None and (self.model is None or self.tokenizer is None):
            self._load_model()
        if self.backend is None:
            self._prepare_model()

    def _load_model(self) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for Qwen direct policy")
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **kwargs)
        model_kwargs = dict(kwargs)
        model_kwargs["torch_dtype"] = _resolve_dtype(self.config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, **model_kwargs)

    def _prepare_model(self) -> None:
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(_runtime_device(self.config.device))

    @property
    def device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                return torch.device(model_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_response(self, prompt: str) -> str:
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        tokenized = _move_tokenized(tokenized, self.device)
        input_len = int(tokenized.get("input_ids", torch.empty(1, 0)).shape[-1])
        with torch.inference_mode():
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=int(self.config.max_new_tokens),
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        if isinstance(generated, torch.Tensor) and generated.ndim == 2 and generated.size(1) > input_len:
            generated = generated[:, input_len:]
        return str(self.tokenizer.decode(generated[0], skip_special_tokens=True)).strip()

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        rows: list[list[float]] = []
        metadata: list[dict[str, Any]] = []
        for state_text, candidates in zip(state_texts, candidate_texts):
            candidate_list = [str(item) for item in candidates]
            if not candidate_list:
                rows.append([])
                metadata.append(
                    {
                        "policy_family": "qwen_direct_admissible",
                        "model_name_or_path": self.config.model_name_or_path,
                        "raw_model_response": "",
                        "parsed_action": "",
                        "parse_status": "no_candidates",
                        "fallback_used": False,
                    }
                )
                continue
            prompt = build_qwen_direct_prompt(
                state_text,
                candidate_list,
                self.config.prompt_template_id,
                tokenizer=self.tokenizer,
                use_chat_template=self.config.use_chat_template and self.backend is None,
                enable_thinking=self.config.enable_thinking,
            )
            if self.backend is not None:
                raw_response = self.backend.generate_text(
                    "You are an ALFWorld agent. Choose one valid text action exactly from AVAILABLE ACTIONS.",
                    prompt,
                )
                backend_metadata = dict(self.backend.metadata()) if hasattr(self.backend, "metadata") else {}
            else:
                raw_response = self._generate_response(prompt)
                backend_metadata = {}
            parsed = parse_admissible_action_response(raw_response, candidate_list, self.config.fallback_strategy)
            chosen_index = candidate_list.index(parsed.action) if parsed.action in candidate_list else 0
            scores = [0.0] * len(candidate_list)
            scores[chosen_index] = 1.0
            rows.append(scores)
            item = {
                "policy_family": "qwen_direct_admissible",
                "model_name_or_path": self.config.model_name_or_path,
                "prompt_template_id": self.config.prompt_template_id,
                "action_selection": "direct_generation_over_admissible_actions",
                "raw_model_response": raw_response,
                "parsed_action": parsed.action,
                "parse_status": parsed.status,
                "fallback_used": bool(parsed.fallback_used),
                "invalid_response": parsed.invalid_response,
                "chosen_index": chosen_index,
            }
            item.update(backend_metadata)
            metadata.append(item)
        self.last_metadata = metadata
        max_width = max((len(row) for row in rows), default=0)
        if max_width == 0:
            return torch.empty(len(rows), 0)
        padded = [row + [torch.finfo(torch.float32).min] * (max_width - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.float32)


def build_qwen_likelihood_context_prompt(
    state_text: str,
    admissible_actions: list[str],
    tokenizer: Any | None = None,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> str:
    action_lines = "\n".join(f"{idx}. {action}" for idx, action in enumerate(admissible_actions, start=1))
    user_content = "\n".join(
        [
            "You are choosing the next ALFWorld text action.",
            "Choose exactly one command from AVAILABLE ACTIONS.",
            "Return exactly one line in this format: ACTION: <exact command>",
            "",
            "[State]",
            str(state_text),
            "",
            "AVAILABLE ACTIONS:",
            action_lines,
            "",
            "ACTION:",
        ]
    )
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "system",
                "content": "You are an ALFWorld agent. Choose one valid text action exactly from AVAILABLE ACTIONS.",
            },
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_content


def _likelihood_context_prompt(state_text: str, admissible_actions: list[str] | None = None) -> str:
    return build_qwen_likelihood_context_prompt(state_text, admissible_actions or [])


class QwenDirectLikelihoodActionScorer(QwenDirectAdmissibleActionScorer):
    """Direct Qwen reference baseline using exact-action conditional likelihood.

    This avoids free-form generation and parsing failures by scoring each admissible
    command as text continuation. It is still not a CLSTR policy.
    """

    def _candidate_loglikelihoods(self, state_text: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        prompt = build_qwen_likelihood_context_prompt(
            state_text,
            candidates,
            tokenizer=self.tokenizer,
            use_chat_template=self.config.use_chat_template,
            enable_thinking=self.config.enable_thinking,
        )
        if prompt.rstrip().endswith("ACTION:"):
            texts = [prompt + " " + str(candidate) for candidate in candidates]
        else:
            texts = [prompt + "ACTION: " + str(candidate) for candidate in candidates]
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").get("input_ids")
        prompt_len = int(prompt_ids.shape[-1]) if isinstance(prompt_ids, torch.Tensor) else 0
        scores_out: list[float] = []
        batch_size = max(1, int(self.config.likelihood_batch_size))
        for start in range(0, len(texts), batch_size):
            tokenized = self.tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True)
            tokenized = _move_tokenized(tokenized, self.device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids))
            with torch.inference_mode():
                output = self.model(**tokenized)
                logits = output.logits.float()
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                labels = input_ids[:, 1:]
                label_mask = attention_mask[:, 1:].bool()
                if prompt_len > 0:
                    positions = torch.arange(labels.size(1), device=labels.device)
                    label_mask = label_mask & (positions >= max(0, prompt_len - 1))
                empty_rows = ~label_mask.any(dim=-1)
                if bool(empty_rows.any()):
                    lengths = attention_mask[:, 1:].sum(dim=-1).clamp_min(1) - 1
                    for row_idx in empty_rows.nonzero(as_tuple=False).flatten().tolist():
                        label_mask[row_idx, int(lengths[row_idx].item())] = True
                gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                masked = torch.where(label_mask, gathered, torch.zeros_like(gathered))
                denom = label_mask.sum(dim=-1).clamp_min(1)
                scores = masked.sum(dim=-1) / denom
            scores_out.extend(float(value) for value in scores.detach().cpu().tolist())
        return scores_out

    def __call__(self, state_texts: list[str], candidate_texts: list[list[str]]) -> torch.Tensor:
        rows: list[list[float]] = []
        metadata: list[dict[str, Any]] = []
        for state_text, candidates in zip(state_texts, candidate_texts):
            candidate_list = [str(item) for item in candidates]
            scores = self._candidate_loglikelihoods(state_text, candidate_list)
            if scores:
                chosen_index = int(torch.argmax(torch.tensor(scores, dtype=torch.float32)).item())
                parsed_action = candidate_list[chosen_index]
                parse_status = "likelihood_ranked_exact_action"
            else:
                chosen_index = 0
                parsed_action = ""
                parse_status = "no_candidates"
            rows.append(scores)
            metadata.append(
                {
                    "policy_family": "qwen_direct_likelihood_admissible",
                    "model_name_or_path": self.config.model_name_or_path,
                    "prompt_template_id": "alfworld_admissible_likelihood_v1",
                    "action_selection": "direct_loglikelihood_ranking_over_admissible_actions",
                    "raw_model_response": "",
                    "parsed_action": parsed_action,
                    "parse_status": parse_status,
                    "fallback_used": False,
                    "invalid_response": None,
                    "chosen_index": chosen_index,
                }
            )
        self.last_metadata = metadata
        max_width = max((len(row) for row in rows), default=0)
        if max_width == 0:
            return torch.empty(len(rows), 0)
        padded = [row + [torch.finfo(torch.float32).min] * (max_width - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.float32)
