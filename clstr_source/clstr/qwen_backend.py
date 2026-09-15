from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QwenBackendConfig:
    model_name_or_path: str = "models/Qwen3-8B"
    cache_dir: str | None = None
    local_files_only: bool = True
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 0.95
    device: str | None = None
    use_chat_template: bool = True
    enable_thinking: bool = False


def _import_torch() -> Any:
    import torch

    return torch


def _import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("transformers is required for QwenGenerationBackend") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    torch = _import_torch()
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


def _runtime_device(config_device: str | None) -> Any:
    if config_device:
        return config_device
    torch = _import_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_tokenized(tokenized: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in tokenized.items()}


class QwenGenerationBackend:
    """Shared Qwen generation backend for CLSTR executor/reference paths.

    This class intentionally centralizes chat-template usage, local-cache loading,
    generation settings, and metadata so AppWorld executor smoke tests and future
    rollout/RL jobs do not drift across multiple Qwen wrappers.
    """

    backend_version = "shared_generation_v1"

    def __init__(
        self,
        config: QwenBackendConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config or QwenBackendConfig()
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            self._load_model()
        self._prepare_model()

    def _load_model(self) -> None:
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
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
    def device(self) -> Any:
        if self.config.device is not None:
            return self.config.device
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                return torch.device(model_device)
        return _runtime_device(None)

    def metadata(self) -> dict[str, Any]:
        return {
            "qwen_backend_version": self.backend_version,
            "model_name_or_path": str(self.config.model_name_or_path),
            "cache_dir": str(self.config.cache_dir) if self.config.cache_dir is not None else None,
            "local_files_only": bool(self.config.local_files_only),
            "torch_dtype": str(self.config.torch_dtype),
            "trust_remote_code": bool(self.config.trust_remote_code),
            "max_new_tokens": int(self.config.max_new_tokens),
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "use_chat_template": bool(self.config.use_chat_template),
            "enable_thinking": bool(self.config.enable_thinking),
        }

    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config.use_chat_template or self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            return str(user_prompt)
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.config.enable_thinking),
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        torch = _import_torch()
        formatted = self.format_chat_prompt(system_prompt, user_prompt)
        tokenized = self.tokenizer(formatted, return_tensors="pt")
        tokenized = _move_tokenized(tokenized, self.device)
        input_ids = tokenized.get("input_ids")
        input_len = int(getattr(input_ids, "shape", (1, 0))[-1]) if input_ids is not None else 0
        do_sample = float(self.config.temperature) > 0.0
        generate_kwargs: dict[str, Any] = {
            **tokenized,
            "max_new_tokens": int(self.config.max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            generate_kwargs["temperature"] = float(self.config.temperature)
            generate_kwargs["top_p"] = float(self.config.top_p)
        with torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)
        if getattr(generated, "ndim", None) == 2 and generated.size(1) > input_len:
            generated = generated[:, input_len:]
        first_sequence = generated[0] if hasattr(generated, "__getitem__") else generated
        return str(self.tokenizer.decode(first_sequence, skip_special_tokens=True)).strip()
