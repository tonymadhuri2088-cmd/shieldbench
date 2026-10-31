import hashlib
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, in_tokens INT, out_tokens INT, cost_usd REAL);
CREATE TABLE IF NOT EXISTS results (
  run_id TEXT, attack_id TEXT, category TEXT, family TEXT, mutation TEXT, language TEXT,
  guard TEXT, target TEXT, benign INT, blocked INT, blocked_by TEXT, response TEXT,
  success INT, judge_reason TEXT, latency_s REAL, in_tokens INT, out_tokens INT,
  cost_usd REAL, cached INT, ts REAL);
CREATE TABLE IF NOT EXISTS adaptive_episodes (
  run_id TEXT, guard TEXT, goal TEXT, trial INT, success INT, rounds INT, refused_rounds INT,
  cost_usd REAL, target TEXT, attacker TEXT, transcript TEXT, ts REAL);
"""


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self.con.executescript(SCHEMA)
        have = {r[1] for r in self.con.execute("PRAGMA table_info(results)")}
        for col, typ in (("attack_text", "TEXT"), ("judge_llm", "INT"), ("judge_llm_reason", "TEXT")):
            if col not in have:  # migrate databases created by v0.1
                self.con.execute(f"ALTER TABLE results ADD COLUMN {col} {typ}")
        have_c = {r[1] for r in self.con.execute("PRAGMA table_info(cache)")}
        if "latency_s" not in have_c:
            self.con.execute("ALTER TABLE cache ADD COLUMN latency_s REAL")
        if "guard_s" not in have:
            self.con.execute("ALTER TABLE results ADD COLUMN guard_s REAL")
        self.con.commit()

    @staticmethod
    def key(model: str, messages: list[dict]) -> str:
        return hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).hexdigest()

    def get_cached(self, key):
        r = self.con.execute("SELECT text,in_tokens,out_tokens,cost_usd,latency_s FROM cache WHERE key=?", (key,)).fetchone()
        return r

    def put_cached(self, key, text, tin, tout, cost, latency=None):
        self.con.execute("INSERT OR REPLACE INTO cache (key,text,in_tokens,out_tokens,cost_usd,latency_s) VALUES (?,?,?,?,?,?)",
                         (key, text, tin, tout, cost, latency))
        self.con.commit()

    def add_result(self, row: dict):
        row = {**row, "ts": time.time()}
        cols = ",".join(row)
        self.con.execute(f"INSERT INTO results ({cols}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        self.con.commit()

    def add_episode(self, row: dict):
        row = {**row, "ts": time.time()}
        self.con.execute(f"INSERT INTO adaptive_episodes ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        self.con.commit()

    def episodes_df(self, run_id: str | None = None):
        import pandas as pd
        q, args = "SELECT * FROM adaptive_episodes", ()
        if run_id:
            q, args = q + " WHERE run_id=?", (run_id,)
        return pd.read_sql_query(q, self.con, params=args)

    def results_df(self, run_id: str | None = None):
        import pandas as pd
        q, args = "SELECT * FROM results", ()
        if run_id:
            q, args = q + " WHERE run_id=?", (run_id,)
        return pd.read_sql_query(q, self.con, params=args)
