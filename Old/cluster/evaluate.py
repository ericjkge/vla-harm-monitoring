#!/usr/bin/env python3
"""Probe evaluation on the generated pairs (leave-one-template-out, same-verb subset, per category/tier)
and train-on-generated -> test-on-benchmark. Writes a plot and prints a table.
  python cluster/evaluate.py --gen $VLA_HARM_ROOT/gen500_records.pkl --bench $VLA_HARM_ROOT/rollouts_bench/records.pkl --out results.png
"""
import argparse, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, collections
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--bench"); ap.add_argument("--out", default="probe_eval.png")
ap.add_argument("--skip_warned", action="store_true"); a = ap.parse_args()
G = pickle.load(open(a.gen, "rb"))
if a.skip_warned: G = [r for r in G if not r.get("warned")]
GS = np.stack([r["hs_safe"] for r in G]); GU = np.stack([r["hs_unsafe"] for r in G]); GD = GU - GS; L = GD.shape[1]
tmpl = np.array([r["template_unsafe"] for r in G]); same_verb = np.array([r["same_verb"] for r in G])
cat = np.array([r["category"] for r in G]); tier = np.array([r["tier"] for r in G])
def lo_group(mask, groups, l):
    c = t = 0
    for g in np.unique(groups[mask]):
        w = GD[groups != g, l].mean(0); test = mask & (groups == g)
        c += int(((GU[test, l] @ w) > (GS[test, l] @ w)).sum()); t += int(test.sum())
    return c / max(t, 1), t
curves = {}
for name, mask in [("leave-one-template-out (all)", np.ones(len(G), bool)), ("same-verb subset", same_verb)]:
    curves[name] = [lo_group(mask, tmpl, l)[0] for l in range(L)]
    b = int(np.argmax(curves[name])); print(f"{name}: n={int(mask.sum())}  best layer {b}  acc {curves[name][b]:.3f}")
best = int(np.argmax(curves["leave-one-template-out (all)"]))
print(f"\nper category at layer {best}:")
for c in sorted(set(cat)): acc, n = lo_group(cat == c, tmpl, best); print(f"  {c:22s} n={n:3d} acc {acc:.2f}")
print(f"per tier at layer {best}:")
for t in sorted(set(tier)): acc, n = lo_group(tier == t, tmpl, best); print(f"  {t} n={n:3d} acc {acc:.2f}")
if a.bench:
    B = pickle.load(open(a.bench, "rb")); by = collections.defaultdict(dict)
    for r in B: by[r["pair_id"]][r["cond"]] = r["hs"]
    tasks = sorted(k for k in by if {"safe", "unsafe"} <= set(by[k]))
    S = np.stack([by[t]["safe"] for t in tasks]); U = np.stack([by[t]["unsafe"] for t in tasks])
    bench = [float(np.mean([(U[i, l] @ GD[:, l].mean(0)) > (S[i, l] @ GD[:, l].mean(0)) for i in range(len(tasks))])) for l in range(L)]
    curves[f"train generated -> test benchmark ({len(tasks)})"] = bench
    print(f"\ngenerated -> benchmark: best layer {int(np.argmax(bench))}  {max(bench)*len(tasks):.0f}/{len(tasks)}")
plt.figure(figsize=(8, 4.5))
for n, c in curves.items(): plt.plot(c, marker="o", ms=3, label=n)
plt.axhline(0.5, ls="--", c="gray"); plt.ylim(0, 1.02); plt.xlabel("layer"); plt.ylabel("fraction ordered correctly"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(a.out, dpi=150); print("plot ->", a.out)
