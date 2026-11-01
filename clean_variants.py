import json, pathlib
p = pathlib.Path("data/generated/variants.json")
rows = json.loads(p.read_text(encoding="utf-8"))
keep = [r for r in rows if not r["id"].startswith("gen-pe-01-")]
p.write_text(json.dumps(keep, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(rows), "->", len(keep))
