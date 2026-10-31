"""The app under test: a fake customer support bot. Each attack is turned into
a message list; indirect attacks hide the payload inside a retrieved 'document'."""
from .fake_data import SYSTEM_PROMPT


def build_messages(user_input: str, document: str | None = None, history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history or []
    if document:
        user_input = (
            "Here is a document retrieved from our knowledge base:\n"
            f"<document>\n{document}\n</document>\n\nUser question: {user_input}"
        )
    msgs.append({"role": "user", "content": user_input})
    return msgs
