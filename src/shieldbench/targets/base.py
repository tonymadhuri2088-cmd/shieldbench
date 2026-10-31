from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Reply:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0


class Target(ABC):
    name: str = "target"

    @abstractmethod
    def generate(self, messages: list[dict]) -> Reply:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]"""
