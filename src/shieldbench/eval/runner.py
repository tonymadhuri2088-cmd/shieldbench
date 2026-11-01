"""Run every attack through every guard on one target. Caches model calls by hash
so reruns cost nothing, and stops at a hard spending cap. Model latency and guard
overhead are recorded separately; cached calls reuse their ORIGINAL latency."""
import time
import uuid
from ..attacks.library import Attack, BENIGN, BENIGN_HARD, load, controls
from ..attacks import mutators
from ..guardrails.base import BLOCK_MESSAGE
from ..guardrails.pipeline import make_guard
from ..judge.rule_judge import judge
from ..targets.support_bot import build_messages
from ..targets.base import Target
from ..store.db import Store
from .cost import CostTracker


def _call(target: Target, store: Store, messages, tracker: CostTracker):
    key = Store.key(target.name, messages)
    hit = store.get_cached(key)
    if hit:
        return hit[0], hit[1], hit[2], 0.0, 1, hit[4]      # original latency (None for v0.1 cache entries)
    t0 = time.perf_counter()
    rep = target.generate(messages)
    lat = time.perf_counter() - t0
    store.put_cached(key, rep.text, rep.in_tokens, rep.out_tokens, rep.cost_usd, lat)
    tracker.add(rep.cost_usd)
    return rep.text, rep.in_tokens, rep.out_tokens, rep.cost_usd, 0, lat


def _timed(fn, text):
    t0 = time.perf_counter()
    v = fn(text)
    wall = time.perf_counter() - t0
    return v, (v.latency_s if v.latency_s is not None else wall)


def _task_ok(marker: str, resp: str) -> bool:
    return any(m and m in resp.lower() for m in marker.lower().split("|"))


def run(target: Target, store: Store, guards=("g0", "g1"), mutations=(), include_benign=True,
        budget_usd=5.0, limit=None, run_id=None, on_row=None, guard_client=None, hard_benign=False, attack_set="main"):
    run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    tracker = CostTracker(budget_usd)
    if attack_set == "heldout":
        from ..attacks.heldout import load_heldout
        base = load_heldout()
    elif attack_set == "heldout2":
        from ..attacks.heldout2 import load_heldout2
        base = load_heldout2()
    elif attack_set == "heldout3":
        from ..attacks.heldout3 import load_heldout3
        base = load_heldout3()
    else:
        base = load()
    attacks: list[Attack] = base + mutators.apply(base, list(mutations))
    if limit:
        attacks = attacks[:limit]
    items = [(a, False) for a in attacks]
    if attack_set in ("heldout", "heldout2", "heldout3"):
        if attack_set == "heldout":
            from ..attacks.heldout import HELDOUT_BENIGN as _hb
        elif attack_set == "heldout3":
            from ..attacks.heldout3 import HELDOUT3_BENIGN as _hb
        else:
            from ..attacks.heldout2 import HELDOUT2_BENIGN as _hb
        items += [(Attack(id=f"hdb-{i:02d}", category="benign_heldout", family="benign_heldout", text=t), True) for i, t in enumerate(_hb)]
    elif include_benign:
        items += [(Attack(id=f"benign-{i:02d}", category="benign", family="benign", text=t), True) for i, t in enumerate(BENIGN)]
        items += [(c, True) for c in controls()]
    if hard_benign:
        items += [(Attack(id=f"hard-{i:02d}", category="benign_hard", family="benign_hard", text=t), True) for i, t in enumerate(BENIGN_HARD)]
    for gkey in guards:
        g = make_guard(gkey, store=store, tracker=tracker, client=guard_client)
        for a, benign in items:
            msgs = build_messages(a.text, a.document, list(a.history))
            vin, g_in = _timed(g.check_input, a.text if not a.document else a.text + "\n" + a.document)
            blocked, by, resp, tin, tout, cost, cached, lat, g_out = bool(vin.blocked), "", "", 0, 0, 0.0, 0, None, 0.0
            if vin.blocked:
                by, resp = f"input:{vin.reason}", BLOCK_MESSAGE
            else:
                resp, tin, tout, cost, cached, lat = _call(target, store, msgs, tracker)
                vout, g_out = _timed(g.check_output, resp)
                if vout.blocked:
                    blocked, by, resp = True, f"output:{vout.reason}", BLOCK_MESSAGE
                elif vout.text is not None and vout.text != resp:
                    resp, by = vout.text, f"output:{vout.reason}"       # redacted, not blocked
            if a.category == "benign_doc":
                ok, why = False, ("task_ok" if (not blocked and _task_ok(a.marker, resp)) else "task_fail")
            else:
                ok, why = (False, "benign") if benign else judge(a, resp, blocked)
            row = dict(run_id=run_id, attack_id=a.id, category=a.category, family=a.family, mutation=a.mutation,
                       language=a.language, guard=g.name, target=target.name, benign=int(benign), blocked=int(blocked),
                       blocked_by=by, response=resp, success=int(ok), judge_reason=why,
                       latency_s=None if lat is None else round(lat, 4), guard_s=round(g_in + g_out, 5),
                       in_tokens=tin, out_tokens=tout, cost_usd=cost, cached=cached,
                       attack_text=a.text + (f"  [DOCUMENT: {a.document}]" if a.document else ""))
            store.add_result(row)
            if on_row:
                on_row(row)
    return run_id, tracker.spent
