"""Provider-independent generation configuration and clients.

Ollama is the default. Gemini dependencies are imported only when selected.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct", "gemini": "gemini-2.5-flash-lite"}
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


@dataclass(frozen=True)
class GenerationSettings:
    provider: str = "ollama"
    model: str = "qwen3:4b-instruct"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 180
    context_tokens: int = 32768
    max_tokens: int = 32
    seed: int = 2026
    temperature: float = 0
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    api_key: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported generation provider: {self.provider}")
        if not self.model:
            raise ValueError("A generation model is required")
        if self.timeout <= 0 or self.max_tokens < 1 or self.context_tokens < 512:
            raise ValueError("Invalid generation timeout or token limits")
        if self.max_tokens + 256 >= self.context_tokens:
            raise ValueError("Output token limit must fit inside context")
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("LLM_BASE_URL must not contain credentials, queries or fragments")
        if self.provider == "gemini" and not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required only for provider=gemini")
        if self.tokenizer and not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision or ""):
            raise ValueError("A custom tokenizer requires an immutable 40-character revision")
        if self.tokenizer_revision and not self.tokenizer:
            raise ValueError("A tokenizer revision requires a tokenizer name")

    def tokenizer_config(self):
        if self.tokenizer:
            return self.tokenizer, self.tokenizer_revision
        if self.model == DEFAULT_MODELS["ollama"]:
            return DEFAULT_TOKENIZER, DEFAULT_TOKENIZER_REVISION
        raise ValueError("Set LLM_TOKENIZER and LLM_TOKENIZER_REVISION for this model")

    def public_config(self):
        return {
            "provider": self.provider, "model": self.model, "temperature": self.temperature,
            "timeout_s": self.timeout,
            **({
                "base_url": self.base_url, "context_tokens": self.context_tokens,
                "max_tokens": self.max_tokens, "seed": self.seed, "thinking": False,
                "tokenizer": self.tokenizer or DEFAULT_TOKENIZER,
                "tokenizer_revision": self.tokenizer_revision or DEFAULT_TOKENIZER_REVISION,
            } if self.provider == "ollama" else {"max_tokens": self.max_tokens}),
        }


def resolve_settings(args=None, api_key=None):
    def option(name, env, fallback):
        value = getattr(args, name, None) if args is not None else None
        return value if value is not None else os.getenv(env, fallback)
    provider = option("provider", "LLM_PROVIDER", "ollama")
    return GenerationSettings(
        provider=provider,
        model=option("model", "LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
        base_url=option("base_url", "LLM_BASE_URL", "http://127.0.0.1:11434"),
        timeout=float(option("timeout", "LLM_TIMEOUT", 180)),
        context_tokens=int(option("context_tokens", "LLM_CONTEXT_TOKENS", 32768)),
        max_tokens=int(option("max_tokens", "LLM_MAX_TOKENS", 32)),
        tokenizer=option("tokenizer", "LLM_TOKENIZER", None),
        tokenizer_revision=option("tokenizer_revision", "LLM_TOKENIZER_REVISION", None),
        api_key=api_key or os.getenv("GOOGLE_API_KEY"),
    )


@dataclass
class Generation:
    content: str


class Generator(Protocol):
    def invoke(self, prompt: str) -> Generation: ...
    def identity(self) -> dict: ...


class OllamaGenerator:
    def __init__(self, settings, opener=urlopen, tokenizer=None):
        self.settings = settings
        self.opener = opener
        self._tokenizer = tokenizer
        self.last_metadata = {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            name, revision = self.settings.tokenizer_config()
            self._tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        return self._tokenizer

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _request(self, endpoint, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.settings.base_url.rstrip("/") + endpoint, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener(request, timeout=self.settings.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP {exc.code}; verify the server and model") from None
        except (URLError, TimeoutError) as exc:
            raise ConnectionError(
                "Cannot reach Ollama; start 'ollama serve' and check LLM_BASE_URL"
            ) from exc

    def identity(self):
        self.settings.tokenizer_config()
        models = self._request("/api/tags").get("models", [])
        requested = self.settings.model
        match = next((m for m in models if m.get("name") in
                      {requested, requested + ":latest"}), None)
        if match is None:
            raise ValueError(f"Model not installed; run: ollama pull {requested}")
        return {"provider": "ollama", "name": match["name"], "digest": match["digest"],
                "server_version": self._request("/api/version").get("version")}

    def invoke(self, prompt):
        self.last_metadata = {}
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        expected_tokens = self.count_tokens(rendered)
        if expected_tokens + self.settings.max_tokens + 1 > self.settings.context_tokens:
            raise ValueError("Prompt exceeds context budget; increase --context-tokens")
        response = self._request("/api/generate", {
            "model": self.settings.model, "stream": False, "raw": True,
            "prompt": rendered,
            "options": {"temperature": self.settings.temperature, "seed": self.settings.seed,
                        "num_ctx": self.settings.context_tokens,
                        "num_predict": self.settings.max_tokens},
        })
        if response.get("done_reason") == "length":
            raise ValueError("Generation reached token limit; increase --max-tokens")
        # Raw rendering makes the local count comparable to the server's count.
        # A mismatch is not silently accepted as a valid experimental observation.
        actual_tokens = response.get("prompt_eval_count")
        if actual_tokens != expected_tokens:
            raise ValueError("Server prompt token count differs; tokenizer mismatch or truncation")
        self.last_metadata = {key: response.get(key) for key in (
            "prompt_eval_count", "eval_count", "load_duration", "total_duration",
        )}
        content = response.get("response")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Generation provider returned no text")
        return Generation(content.strip())


class GeminiGenerator:
    def __init__(self, settings):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Install requirements-gemini.txt to use provider=gemini") from None
        self.settings = settings
        self.client = ChatGoogleGenerativeAI(
            model=settings.model, google_api_key=settings.api_key,
            temperature=settings.temperature, max_output_tokens=settings.max_tokens,
            timeout=settings.timeout,
        )

    def identity(self):
        return {"provider": "gemini", "name": self.settings.model,
                "note": "Provider alias; immutable model revision unavailable"}

    def count_tokens(self, text):
        return self.client.get_num_tokens(text)

    def invoke(self, prompt):
        response = self.client.invoke(prompt)
        if not isinstance(response.content, str) or not response.content.strip():
            raise ValueError("Generation provider returned no text")
        return Generation(response.content.strip())


def create_generator(settings):
    return OllamaGenerator(settings) if settings.provider == "ollama" else GeminiGenerator(settings)
