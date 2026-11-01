"""Target for ANY model that speaks the OpenAI chat format (Ollama's /v1, LM Studio, vLLM, llama.cpp server, hosted models).
It lets ShieldBench attack a model of your choice. Settings come from environment variables (or .env):
    OPENAI_COMPAT_URL    base address, default http://localhost:11434/v1   (Ollama)
    OPENAI_COMPAT_MODEL  model name, default the Ollama model in config
    OPENAI_COMPAT_KEY    optional key, sent as 'Authorization: Bearer ...' (never printed or saved)
    OPENAI_COMPAT_PRICE  optional 'input,output' USD per million tokens, so the spending cap works for paid models
Example:  python -m shieldbench.run --target openai --guards g0,g2 --attack-set heldout4 --limit 10"""
import os
import requests
from .base import Target, Reply
from .. import config


class OpenAICompatTarget(Target):
    def __init__(self, base_url: str | None = None, model: str | None = None, key: str | None = None, price: str | None = None):
        self.base = (base_url or os.getenv("OPENAI_COMPAT_URL") or config.OLLAMA_URL.rstrip("/") + "/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_COMPAT_MODEL") or config.OLLAMA_MODEL
        self._key = key if key is not None else os.getenv("OPENAI_COMPAT_KEY", "")
        p = price or os.getenv("OPENAI_COMPAT_PRICE", "")
        try:
            self.price = tuple(float(x) for x in p.split(",")) if p else (0.0, 0.0)
        except ValueError:
            self.price = (0.0, 0.0)
        self.name = f"openai:{self.model}"

    def generate(self, messages):
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        try:
            r = requests.post(f"{self.base}/chat/completions", headers=headers, timeout=300,
                              json={"model": self.model, "messages": messages, "temperature": 0, "seed": config.SEED, "max_tokens": 300, "stream": False})
            r.raise_for_status()
        except requests.HTTPError as e:      # say what failed, never echo the key or the request
            raise RuntimeError(f"the model server answered HTTP {e.response.status_code} at {self.base}") from None
        except requests.RequestException as e:
            raise RuntimeError(f"cannot reach the model server at {self.base} ({type(e).__name__})") from None
        d = r.json()
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        u = d.get("usage") or {}
        i, o = int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
        return Reply(text=text, in_tokens=i, out_tokens=o, cost_usd=(i * self.price[0] + o * self.price[1]) / 1e6)
