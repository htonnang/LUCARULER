"""Soundness and screening power of hybrid/constraints.py on tasks with known test outputs.
python3 -I measure_constraints.py TASK_DIR [--luca-runs DIR]   (aggregate numbers only)"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import constraints as C
import solve_hybrid as SH
from collections import Counter

tdir = sys.argv[1]; luca = sys.argv[sys.argv.index("--luca-runs") + 1] if "--luca-runs" in sys.argv else None
rnd = random.Random(0)
n = active = sound = sound_old = 0
rule_use, rule_fail = Counter(), Counter()
neg = {k: [0, 0, 0] for k in ("identity", "1-3 cells changed", "luca wrong answer")}   # [n, old rejects, new rejects]

def perturb(g):
    g = [list(r) for r in g]; cols = list({v for r in g for v in r} | {0, rnd.randrange(10)})
    for _ in range(rnd.randint(1, 3)):
        i, j = rnd.randrange(len(g)), rnd.randrange(len(g[0]))
        g[i][j] = rnd.choice([c for c in cols if c != g[i][j]] or [(g[i][j] + 1) % 10])
    return g

for f in sorted(os.listdir(tdir)):
    if not f.endswith(".json") or f.endswith(".meta.json"):
        continue
    t = json.load(open(os.path.join(tdir, f)))
    rules = C.fit_rules(t); sizes, pal = SH.size_rule(t), SH.palette_rule(t)
    for k, tt in enumerate(t["test"]):
        x, y = tt["input"], tt["output"]
        n += 1; active += bool(rules)
        for name, _ in rules:
            rule_use[name.split(":")[0] if not name.startswith("size=") else "size(any)"] += 1
        b = C.broken(rules, x, y)
        sound += not b
        for name in b:
            rule_fail[name.split(":")[0] if not name.startswith("size=") else "size(any)"] += 1
        sound_old += SH.consistent(y, x, sizes, pal)
        cands = [("identity", x), ("1-3 cells changed", perturb(y))]
        if luca and os.path.exists(os.path.join(luca, f)):
            try:
                r = json.load(open(os.path.join(luca, f)))[k]
                cands += [("luca wrong answer", p) for p in r.get("predictions", [])[:2] if p != y]
            except Exception:
                pass
        for kind, g in cands:
            if g == y:
                continue
            neg[kind][0] += 1
            neg[kind][1] += not SH.consistent(g, x, sizes, pal)
            neg[kind][2] += not C.passes(rules, x, g)

print(f"test inputs {n} | with >=1 rule {active}")
print(f"TRUE answer kept: new rules {sound}/{n} ({100*sound/n:.1f}%)   old rules {sound_old}/{n} ({100*sound_old/n:.1f}%)")
print("wrong candidates rejected (old -> new):")
for k, (m, o, nw) in neg.items():
    if m:
        print(f"  {k:<20} {m:5d}   {100*o/m:5.1f}% -> {100*nw/m:5.1f}%")
print("rule use:", dict(rule_use.most_common()))
print("rule failures on true answers:", dict(rule_fail.most_common()))
