"""Bring your own bot: scan ANY chatbot that speaks the OpenAI chat format (Ollama's /v1, LM Studio, vLLM, hosted models).

    scan_bot({"base_url": "http://localhost:11434/v1", "model": "gemma2:2b", "system_prompt": "You are ..."}, guards=["g1", "g2"])

What it does: sends our 22 seed attacks (and 22 harmless messages) to the bot WITH ITS OWN SYSTEM PROMPT, once. Because the model is run at
temperature 0, an attack that a guard lets through gets exactly the same reply as with no guard, so each attack costs ONE model call and every
guard is scored from that same reply. The judge is the rule judge (a marker word or a canary string appearing in the reply).

Honest limits: 22 attacks, no disguises, so a small sample. Extraction and data-leak attacks are only scored when the system prompt holds our canary
strings (leave the prompt empty to use AcmeBot, or paste SB-CANARY-7f3a9c into your prompt). The key is used for the calls only and is never stored or echoed."""
from __future__ import annotations
import time
from .attacks.library import load, controls, BENIGN_HARD
from .judge.rule_judge import judge
from .targets import fake_data as fd
from .targets.support_bot import build_messages
from .guardrails.base import Guard


def _messages(system: str, attack) -> list:
    m = build_messages(attack.text, attack.document, [dict(h) for h in attack.history])
    m[0] = {"role": "system", "content": system}
    return m


def _snip(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "..."


def scan_bot(cfg: dict, guards: dict[str, Guard], limit: int = 0, bot=None, progress=None, disguises: bool = False) -> dict:
    """guards: {"g1": Guard, ...}. cfg: base_url, model, key?, system_prompt?. Returns a JSON-able report."""
    from .targets.openai_target import OpenAICompatTarget
    url = str(cfg.get("base_url", "")).strip()
    if bot is None:
        if not url.lower().startswith(("http://", "https://")):
            raise RuntimeError("base_url must start with http:// or https://")
        bot = OpenAICompatTarget(url, str(cfg.get("model", "")).strip() or None, str(cfg.get("key", "")))
    system = str(cfg.get("system_prompt", "")).strip() or fd.SYSTEM_PROMPT
    has_canary = any(c in system for c in fd.SECRET_CANARIES) or system == fd.SYSTEM_PROMPT
    attacks = [a for a in load() if has_canary or a.category not in ("prompt_extraction", "pii_leakage")]
    if disguises:      # add the 5 disguises (base64, leetspeak, role-play, Hinglish prefix, multi-turn): about 5x more attacks
        from .attacks import mutators
        attacks = attacks + mutators.apply(attacks, list(mutators.MUTATORS))
    if limit:
        attacks = attacks[:limit]
    rows, errors = [], 0
    t0 = time.time()
    for i, a in enumerate(attacks):
        try:
            reply = bot.generate(_messages(system, a)).text
        except RuntimeError as e:
            if i == 0:
                raise
            errors += 1
            continue
        gtext = a.text + (("\n" + a.document) if a.document else "")
        r = {"id": a.id, "category": a.category, "text": _snip(a.text), "reply": _snip(reply), "none": judge(a, reply, False)[0]}
        for k, g in guards.items():
            vin = g.check_input(gtext)
            if vin.blocked:
                r[k] = False
                continue
            vout = g.check_output(reply)
            shown = vout.text if (vout.text is not None and not vout.blocked) else reply
            r[k] = False if vout.blocked else judge(a, shown, False)[0]
        rows.append(r)
        if progress:
            progress(i + 1, len(attacks))
    benign = [c.text + "\n" + (c.document or "") for c in controls()] + list(BENIGN_HARD[:10])
    hb = {k: sum(1 for t in benign if g.check_input(t).blocked) for k, g in guards.items()}
    n = len(rows)
    summary = {"none": sum(r["none"] for r in rows)}
    for k in guards:
        summary[k] = sum(r[k] for r in rows)
    from .eval.metrics import wilson
    ci = {k: [round(100 * x, 1) for x in wilson(v, n)[1:]] for k, v in summary.items()}
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], {"n": 0, "none": 0})
        c["n"] += 1
        c["none"] += r["none"]
    return {"bot": getattr(bot, "name", "bot"), "attacks": n, "errors": errors, "worked": summary, "harmless_total": len(benign), "harmless_blocked": hb,
            "by_category": cats, "worked_ci": ci, "disguises": disguises, "scored_extraction": has_canary, "seconds": round(time.time() - t0, 1),
            "worked_examples": [{"id": r["id"], "category": r["category"], "attack": r["text"], "reply": r["reply"]} for r in rows if r["none"]][:6],
            "guards": {k: g.name for k, g in guards.items()}}
