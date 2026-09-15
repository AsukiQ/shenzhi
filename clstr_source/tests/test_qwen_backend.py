import importlib
import sys


class _FakeTensor:
    shape = (1, 3)

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class _FakeGenerated:
    ndim = 2

    def __init__(self):
        self.items = []

    def size(self, dim):
        return 5 if dim == 1 else 1

    def __getitem__(self, item):
        self.items.append(item)
        return self


class _FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    @staticmethod
    def inference_mode():
        return _FakeInferenceMode()


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, response: str = "print('ok')", accepts_enable_thinking: bool = True):
        self.response = response
        self.accepts_enable_thinking = accepts_enable_thinking
        self.chat_calls = []
        self.last_text = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if not self.accepts_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("enable_thinking is not supported")
        self.chat_calls.append((messages, tokenize, add_generation_prompt, kwargs))
        return "CHAT::" + messages[-1]["content"]

    def __call__(self, text, return_tensors="pt", **kwargs):
        del return_tensors, kwargs
        self.last_text = text
        return {
            "input_ids": _FakeTensor(),
            "attention_mask": _FakeTensor(),
        }

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return self.response


class _FakeParam:
    device = "cpu"


class _FakeModel:
    def __init__(self):
        self.generate_kwargs = None
        self.moved_to = None

    @property
    def device(self):
        return "cpu"

    def parameters(self):
        return iter([_FakeParam()])

    def eval(self):
        return None

    def to(self, device):
        self.moved_to = device
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return _FakeGenerated()


def _backend_module():
    return importlib.import_module("clstr.qwen_backend")


def _build_backend(monkeypatch, *, tokenizer=None, config=None):
    backend_module = _backend_module()
    monkeypatch.setattr(backend_module, "_import_torch", lambda: _FakeTorch)
    return backend_module.QwenGenerationBackend(
        config=config or backend_module.QwenBackendConfig(model_name_or_path="models/Qwen3-8B", device="cpu"),
        model=_FakeModel(),
        tokenizer=tokenizer or _FakeTokenizer(),
    )


def test_qwen_backend_import_does_not_require_torch(monkeypatch):
    original_module = sys.modules.pop("clstr.qwen_backend", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    try:
        backend_module = importlib.import_module("clstr.qwen_backend")
        assert backend_module.QwenBackendConfig(model_name_or_path="x").model_name_or_path == "x"
    finally:
        sys.modules.pop("clstr.qwen_backend", None)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        if original_module is not None:
            sys.modules["clstr.qwen_backend"] = original_module


def test_qwen_backend_formats_chat_template_with_thinking_disabled(monkeypatch):
    tokenizer = _FakeTokenizer()
    backend = _build_backend(
        monkeypatch,
        tokenizer=tokenizer,
        config=_backend_module().QwenBackendConfig(model_name_or_path="models/Qwen3-8B", max_new_tokens=7, device="cpu"),
    )

    prompt = backend.format_chat_prompt("system text", "user text")

    assert prompt == "CHAT::user text"
    assert tokenizer.chat_calls[0][0][0]["role"] == "system"
    assert tokenizer.chat_calls[0][0][0]["content"] == "system text"
    assert tokenizer.chat_calls[0][3]["enable_thinking"] is False


def test_qwen_backend_falls_back_when_chat_template_lacks_enable_thinking(monkeypatch):
    tokenizer = _FakeTokenizer(accepts_enable_thinking=False)
    backend = _build_backend(monkeypatch, tokenizer=tokenizer)

    prompt = backend.format_chat_prompt("system text", "user text")

    assert prompt == "CHAT::user text"
    assert tokenizer.chat_calls[0][3] == {}


def test_qwen_backend_generate_strips_prompt_tokens_and_records_metadata(monkeypatch):
    backend_module = _backend_module()
    model = _FakeModel()
    tokenizer = _FakeTokenizer(response="CODE")
    monkeypatch.setattr(backend_module, "_import_torch", lambda: _FakeTorch)
    backend = backend_module.QwenGenerationBackend(
        config=backend_module.QwenBackendConfig(
            model_name_or_path="models/Qwen3-8B",
            max_new_tokens=9,
            temperature=0.0,
            local_files_only=True,
            device="cpu",
        ),
        model=model,
        tokenizer=tokenizer,
    )

    text = backend.generate_text("sys", "write code")
    metadata = backend.metadata()

    assert text == "CODE"
    assert model.generate_kwargs["max_new_tokens"] == 9
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["input_ids"].moved_to == "cpu"
    assert tokenizer.last_text == "CHAT::write code"
    assert metadata["qwen_backend_version"] == "shared_generation_v1"
    assert metadata["model_name_or_path"] == "models/Qwen3-8B"
    assert metadata["local_files_only"] is True
    assert metadata["enable_thinking"] is False
