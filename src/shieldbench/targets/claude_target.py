import os
from .base import Target, Reply
from .. import config


class ClaudeTarget(Target):
    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so local-only use needs no key
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY in .env to use the Claude target.")
        self.client = anthropic.Anthropic()
        self.model = model or config.CLAUDE_TARGET_MODEL
        self.name = f"claude:{self.model}"

    def generate(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        resp = self.client.messages.create(
            model=self.model, max_tokens=300, system=system, messages=chat
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        pin, pout = config.PRICES_PER_MTOK.get(self.model, (0.0, 0.0))
        u = resp.usage
        cost = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
        return Reply(text, u.input_tokens, u.output_tokens, cost)
