"""Central settings. Everything secret comes from environment variables."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
DB_PATH = Path(os.getenv("SHIELDBENCH_DB", ROOT / "results" / "shieldbench.db"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
CLAUDE_TARGET_MODEL = os.getenv("CLAUDE_TARGET_MODEL", "claude-haiku-5-5")
BUDGET_USD = float(os.getenv("BUDGET_USD", "5"))

# USD per million tokens (input, output). Re-check before large runs.
# Source: platform.claude.com pricing page, checked 2026-10-31. Haiku prompts
# over 100k tokens cost more; we never send prompts that long.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
}
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
SEED = 1234
