import requests
from .base import Target, Reply
from .. import config


class OllamaTarget(Target):
    def __init__(self, model: str | None = None, url: str | None = None, num_ctx: int = 2048):
        self.model = model or config.OLLAMA_MODEL
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.num_ctx = num_ctx          # 2048 keeps more of a 4 GB GPU free
        self.name = f"ollama:{self.model}"

    def generate(self, messages):
        r = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "seed": config.SEED, "num_ctx": self.num_ctx, "num_predict": 300},
            },
            timeout=300,
        )
        r.raise_for_status()
        d = r.json()
        return Reply(
            text=d["message"]["content"],
            in_tokens=d.get("prompt_eval_count", 0),
            out_tokens=d.get("eval_count", 0),
            cost_usd=0.0,  # local model: free
        )
