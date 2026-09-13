import io
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from src.generation import (
    GenerationSettings,
    OllamaGenerator,
    create_generator,
    resolve_settings,
)


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def test_local_is_default_without_google_credentials(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = resolve_settings()
    assert settings.provider == "ollama"
    assert settings.model == "qwen3:4b-instruct"
    assert isinstance(create_generator(settings), OllamaGenerator)
    assert "api_key" not in settings.public_config()


def test_cli_overrides_environment_and_secrets_are_not_serialized(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "old")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    settings = resolve_settings(SimpleNamespace(provider="ollama", model="new"))
    assert settings.model == "new"
    assert "secret" not in repr(settings)
    assert "secret" not in json.dumps(settings.public_config())


@pytest.mark.parametrize("kwargs", [
    {"provider": "unknown"}, {"model": ""}, {"timeout": 0},
    {"context_tokens": 1}, {"max_tokens": 0}, {"max_tokens": 32700},
    {"base_url": "file:///tmp/model"},
    {"base_url": "http://user:password@localhost:11434"},
    {"provider": "gemini"},
])
def test_invalid_settings_fail_before_requests(kwargs):
    with pytest.raises(ValueError):
        GenerationSettings(**kwargs)


def test_ollama_request_returns_text_and_pins_generation_parameters():
    requests = []
    def opener(request, timeout):
        requests.append((request, timeout))
        return response({"message": {"content": " B "}, "done_reason": "stop"})
    client = OllamaGenerator(GenerationSettings(), opener=opener)
    assert client.invoke("question").content == "B"
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_ctx"] == 32768
    assert timeout == 180


def test_ollama_identity_records_digest():
    replies = iter([
        {"models": [{"name": "qwen3:4b-instruct", "digest": "immutable-hash"}]},
        {"version": "example-version"},
    ])
    client = OllamaGenerator(GenerationSettings(), opener=lambda *a, **k: response(next(replies)))
    assert client.identity()["digest"] == "immutable-hash"


def test_missing_model_gives_actionable_error():
    client = OllamaGenerator(GenerationSettings(), opener=lambda *a, **k: response({"models": []}))
    with pytest.raises(ValueError, match="ollama pull"):
        client.identity()


@pytest.mark.parametrize("result", [
    {"message": {"content": ""}}, {"message": {"content": []}},
    {"message": {"content": "B"}, "done_reason": "length"},
])
def test_malformed_or_truncated_generation_is_not_scored(result):
    client = OllamaGenerator(GenerationSettings(), opener=lambda *a, **k: response(result))
    with pytest.raises(ValueError):
        client.invoke("question")


def test_oversized_prompt_fails_without_calling_server():
    client = OllamaGenerator(
        GenerationSettings(context_tokens=512),
        opener=lambda *a, **k: pytest.fail("must not send truncated prompt"),
    )
    with pytest.raises(ValueError, match="context"):
        client.invoke("a" * 513)


@pytest.mark.parametrize("exception,expected", [
    (URLError("offline"), ConnectionError),
    (HTTPError("url", 404, "sensitive", {}, None), RuntimeError),
])
def test_transport_errors_are_actionable(exception, expected):
    def fail(*a, **k):
        raise exception
    client = OllamaGenerator(GenerationSettings(), opener=fail)
    with pytest.raises(expected) as caught:
        client.invoke("question")
    assert "sensitive" not in str(caught.value)


def test_gemini_is_lazy_and_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    assert isinstance(create_generator(GenerationSettings()), OllamaGenerator)
    with pytest.raises(ImportError, match="requirements-gemini"):
        create_generator(GenerationSettings(provider="gemini", model="cloud", api_key="test"))


def test_optional_cloud_adapter_uses_selected_model(monkeypatch):
    received = {}
    def factory(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(invoke=lambda p: SimpleNamespace(content="C"))
    monkeypatch.setitem(sys.modules, "langchain_google_genai",
                        SimpleNamespace(ChatGoogleGenerativeAI=factory))
    client = create_generator(
        GenerationSettings(provider="gemini", model="chosen-model", api_key="secret")
    )
    assert client.invoke("question").content == "C"
    assert received["model"] == "chosen-model"
    assert client.identity()["name"] == "chosen-model"
    assert "secret" not in json.dumps(client.identity())
