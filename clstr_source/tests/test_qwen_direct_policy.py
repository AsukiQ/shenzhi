import torch

from clstr.qwen_direct_policy import (
    QwenDirectPolicyConfig,
    QwenDirectAdmissibleActionScorer,
    QwenDirectLikelihoodActionScorer,
    build_qwen_direct_prompt,
    build_qwen_likelihood_context_prompt,
    parse_admissible_action_response,
)


def test_parse_admissible_action_response_exact_and_normalized_matches():
    candidates = ["look", "open fridge", "take apple 1 from fridge 1"]

    exact = parse_admissible_action_response("Action: open fridge", candidates)
    normalized = parse_admissible_action_response("OPEN   FRIDGE.", candidates)

    assert exact.action == "open fridge"
    assert exact.status == "exact_match"
    assert normalized.action == "open fridge"
    assert normalized.status == "normalized_match"


def test_parse_admissible_action_response_invalid_uses_explicit_fallback():
    result = parse_admissible_action_response(
        "I would inspect the sink.",
        ["go to cabinet 1", "look"],
        fallback_strategy="look_if_available_else_first",
    )

    assert result.action == "look"
    assert result.status == "fallback"
    assert result.fallback_used is True
    assert result.invalid_response == "I would inspect the sink."


def test_parse_admissible_action_response_loose_number_text_is_fallback_not_index_match():
    result = parse_admissible_action_response(
        "maybe action 2 because the fridge matters",
        ["look", "open fridge"],
        fallback_strategy="look_if_available_else_first",
    )

    assert result.action == "look"
    assert result.status == "fallback"
    assert result.fallback_used is True


def test_parse_admissible_action_response_accepts_numbered_action_line_and_late_action_tag():
    candidates = ["look", "open cabinet 1", "take apple 1 from countertop 1"]

    numbered = parse_admissible_action_response("2. open cabinet 1", candidates)
    tagged = parse_admissible_action_response(
        "I need the object first.\nACTION: take apple 1 from countertop 1",
        candidates,
    )

    assert numbered.action == "open cabinet 1"
    assert numbered.status == "numbered_action_match"
    assert tagged.action == "take apple 1 from countertop 1"
    assert tagged.status == "exact_match"


def test_qwen_direct_prompt_can_use_chat_template_and_available_actions_label():
    class _ChatTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            self.calls.append((messages, tokenize, add_generation_prompt, kwargs))
            return "CHAT::" + messages[-1]["content"]

    tokenizer = _ChatTokenizer()
    prompt = build_qwen_direct_prompt(
        "goal: heat apple\nobservation: kitchen",
        ["look", "open fridge 1"],
        tokenizer=tokenizer,
        use_chat_template=True,
        enable_thinking=False,
    )

    assert prompt.startswith("CHAT::")
    assert "AVAILABLE ACTIONS" in prompt
    assert "1. look" in prompt
    assert "2. open fridge 1" in prompt
    assert tokenizer.calls[0][3]["enable_thinking"] is False


def test_qwen_likelihood_context_prompt_includes_available_actions():
    prompt = build_qwen_likelihood_context_prompt(
        "goal: heat apple\nobservation: kitchen",
        ["look", "open fridge 1"],
    )

    assert "AVAILABLE ACTIONS" in prompt
    assert "1. look" in prompt
    assert "2. open fridge 1" in prompt
    assert prompt.rstrip().endswith("ACTION:")


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = ""

    def __call__(self, text, return_tensors="pt", **kwargs):
        del return_tensors, kwargs
        self.last_prompt = text
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return self.response


class _FakeSharedBackend:
    def __init__(self):
        self.calls = []

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "ACTION: open fridge"

    def metadata(self) -> dict:
        return {
            "qwen_backend_version": "shared_generation_v1",
            "model_name_or_path": "models/Qwen3-8B",
            "local_files_only": True,
            "enable_thinking": False,
        }


def test_qwen_direct_scorer_can_use_shared_backend():
    backend = _FakeSharedBackend()
    scorer = QwenDirectAdmissibleActionScorer(
        config=QwenDirectPolicyConfig(model_name_or_path="models/Qwen3-8B"),
        backend=backend,
    )

    scores = scorer(["goal: cool apple"], [["look", "open fridge"]])

    assert scores.tolist() == [[0.0, 1.0]]
    assert backend.calls[0][0] == "You are an ALFWorld agent. Choose one valid text action exactly from AVAILABLE ACTIONS."
    assert "AVAILABLE ACTIONS" in backend.calls[0][1]
    assert scorer.last_metadata[0]["qwen_backend_version"] == "shared_generation_v1"
    assert scorer.last_metadata[0]["model_name_or_path"] == "models/Qwen3-8B"
    assert scorer.last_metadata[0]["parse_status"] == "exact_match"


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.moved_to = None

    @property
    def device(self):
        return self.anchor.device

    def to(self, device):
        self.moved_to = torch.device(device)
        return self

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 8
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


def test_qwen_direct_scorer_returns_scores_and_metadata_without_training():
    tokenizer = _FakeTokenizer("Action: open fridge")
    scorer = QwenDirectAdmissibleActionScorer(
        config=QwenDirectPolicyConfig(
            model_name_or_path="fake-qwen",
            max_new_tokens=8,
            fallback_strategy="first_admissible",
        ),
        model=_FakeModel(),
        tokenizer=tokenizer,
    )

    scores = scorer(
        ["goal: cool apple\nobservation: kitchen\nhistory: <empty>"],
        [["look", "open fridge"]],
    )

    assert scores.tolist() == [[0.0, 1.0]]
    assert scorer.last_metadata[0]["policy_family"] == "qwen_direct_admissible"
    assert scorer.last_metadata[0]["model_name_or_path"] == "fake-qwen"
    assert scorer.last_metadata[0]["raw_model_response"] == "Action: open fridge"
    assert scorer.last_metadata[0]["parse_status"] == "exact_match"
    assert scorer.last_metadata[0]["fallback_used"] is False
    assert "admissible commands" in tokenizer.last_prompt.lower()


def test_qwen_direct_scorer_auto_moves_model_to_cuda_when_available(monkeypatch):
    monkeypatch.setattr("clstr.qwen_direct_policy.torch.cuda.is_available", lambda: True)
    model = _FakeModel()

    QwenDirectAdmissibleActionScorer(
        config=QwenDirectPolicyConfig(
            model_name_or_path="fake-qwen",
            max_new_tokens=8,
        ),
        model=model,
        tokenizer=_FakeTokenizer("Action: look"),
    )

    assert model.moved_to == torch.device("cuda")


class _LikelihoodTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, return_tensors="pt", padding=False, truncation=False, **kwargs):
        del return_tensors, padding, truncation, kwargs
        if isinstance(texts, str):
            texts = [texts]
        width = 6
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention_mask = torch.ones(len(texts), width, dtype=torch.long)
        for row, text in enumerate(texts):
            if text.endswith("open fridge"):
                input_ids[row, -1] = 4
            elif text.endswith("look"):
                input_ids[row, -1] = 3
            else:
                input_ids[row, -1] = 2
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _LikelihoodModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def to(self, device):
        return self

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        vocab = 8
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab, dtype=torch.float32)
        logits[:, :-1, 4] = 5.0
        logits[:, :-1, 3] = -1.0
        return type("Out", (), {"logits": logits})


def test_qwen_direct_likelihood_scorer_ranks_exact_admissible_actions_without_generation():
    scorer = QwenDirectLikelihoodActionScorer(
        config=QwenDirectPolicyConfig(model_name_or_path="fake-qwen"),
        model=_LikelihoodModel(),
        tokenizer=_LikelihoodTokenizer(),
    )

    scores = scorer(
        ["goal: cool apple\nobservation: kitchen\nhistory: <empty>"],
        [["look", "open fridge"]],
    )

    assert int(torch.argmax(scores, dim=-1).item()) == 1
    assert scorer.last_metadata[0]["policy_family"] == "qwen_direct_likelihood_admissible"
    assert scorer.last_metadata[0]["action_selection"] == "direct_loglikelihood_ranking_over_admissible_actions"
    assert scorer.last_metadata[0]["fallback_used"] is False


def test_qwen_direct_likelihood_scorer_batches_candidates_to_avoid_oom():
    class _RecordingTokenizer(_LikelihoodTokenizer):
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, texts, *args, **kwargs):
            if isinstance(texts, list):
                self.batch_sizes.append(len(texts))
            return super().__call__(texts, *args, **kwargs)

    tokenizer = _RecordingTokenizer()
    scorer = QwenDirectLikelihoodActionScorer(
        config=QwenDirectPolicyConfig(model_name_or_path="fake-qwen", likelihood_batch_size=1),
        model=_LikelihoodModel(),
        tokenizer=tokenizer,
    )

    scorer(["state"], [["look", "open fridge"]])

    assert max(tokenizer.batch_sizes) <= 1


def test_qwen_direct_likelihood_chat_template_scores_action_tag_continuations():
    class _ChatLikelihoodTokenizer(_LikelihoodTokenizer):
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __init__(self):
            self.texts = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            del tokenize, add_generation_prompt, kwargs
            return "<chat>" + messages[-1]["content"] + "\n<assistant>\n"

        def __call__(self, texts, *args, **kwargs):
            if isinstance(texts, list):
                self.texts.extend(texts)
            else:
                self.texts.append(texts)
            return super().__call__(texts, *args, **kwargs)

    tokenizer = _ChatLikelihoodTokenizer()
    scorer = QwenDirectLikelihoodActionScorer(
        config=QwenDirectPolicyConfig(model_name_or_path="fake-qwen", use_chat_template=True),
        model=_LikelihoodModel(),
        tokenizer=tokenizer,
    )

    scorer(["goal: cool apple\nobservation: kitchen"], [["look", "open fridge"]])

    joined = "\n".join(tokenizer.texts)
    assert "AVAILABLE ACTIONS" in joined
    assert "ACTION: open fridge" in joined
