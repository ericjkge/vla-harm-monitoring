"""Layer-wise linear probes on activations saved by scripts/extract_activations.py (CPU only; numpy+scipy+matplotlib).

For each layer: standardise on train, fit L2 logistic regression, pick C on val (AUROC), report test.
Reported per layer: accuracy, AUROC, pairwise accuracy (score(harmful) > score(harmless) within a pair),
with group-bootstrap 95% CIs on test accuracy. Controls: shuffled train labels, pair-difference probe,
and subset breakdowns (non-smash, scene-only, text-distinct, high label confidence).

  python scripts/train_probes.py --acts paligemma=cache/acts/paligemma.npz --feature last_instr
  python scripts/train_probes.py --acts pg=cache/acts/paligemma.npz base=cache/acts/pi05_base.npz \
        ft=cache/acts/pi05_libero.npz --feature mean_instr --out results/probes
Outputs: results/probes/{feature}.json, {feature}_layers.png (test accuracy vs layer, per model, with baselines).
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def fit_logreg(X, y, C):
    """L2 logistic regression (intercept unpenalised), L-BFGS. y in {0,1}."""
    n, d = X.shape
    s = 2 * y - 1.0

    def f(w):
        z = s * (X @ w[:d] + w[d])
        loss = np.logaddexp(0, -z).sum() * C + 0.5 * w[:d] @ w[:d]
        p = 1 / (1 + np.exp(np.clip(z, -50, 50)))  # sigmoid(-z)
        g = -(X * (s * p)[:, None]).sum(0) * C + w[:d]
        return loss, np.append(g, -(s * p).sum() * C)

    return minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_meta():
    """per-sample metadata aligned to probe_table rows (pair_id, label) -> dict."""
    pairs = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "libero_safety/minimal_pairs_500.jsonl")}
    tab = {(r["pair_id"], r["label"]): r for r in map(json.loads, open(ROOT / "results/dataset_review/probe_table.jsonl"))}
    return pairs, tab


def pair_index(pair_ids, label):
    """for each sample, index of its partner (same pair_id, other label)."""
    idx = {(p, l): i for i, (p, l) in enumerate(zip(pair_ids, label))}
    return np.array([idx.get((p, 1 - l), -1) for p, l in zip(pair_ids, label)])


def boot_ci(correct, groups, rng, B=1000):
    ug = np.unique(groups)
    by = {g: correct[groups == g] for g in ug}
    m = [np.concatenate([by[g] for g in rng.choice(ug, len(ug))]).mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def probe_layer(Xtr, ytr, Xva, yva, Xte, shuffle_rng=None):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    if shuffle_rng is not None:  # full null: C is tuned on shuffled val labels too
        ytr, yva = shuffle_rng.permutation(ytr), shuffle_rng.permutation(yva)
    best = None
    for C in CS:
        w = fit_logreg(Xtr, ytr, C)
        a = auroc(Xva @ w[:-1] + w[-1], yva)
        if best is None or a > best[0]: best = (a, C, w)
    _, C, w = best
    return C, Xte @ w[:-1] + w[-1], Xva @ w[:-1] + w[-1]


def diff_probe_layer(X, y, pid, splits, Xte_idx, rng):
    """probe trained on pair differences (harmful - harmless), +d labelled 1 and -d labelled 0, no intercept effect."""
    def diffs(sp):
        m = (splits == sp) & (y == 1)
        i = np.where(m)[0]
        j = np.array([np.where((pid == pid[k]) & (y == 0))[0][0] for k in i])
        return X[i] - X[j], i, j
    Dtr, _, _ = diffs("train"); Dva, _, _ = diffs("val"); Dte, i, j = diffs("test")
    sd = np.concatenate([Dtr, -Dtr]).std(0) + 1e-6
    best = None
    for C in CS:
        w = fit_logreg(np.concatenate([Dtr, -Dtr]) / sd, np.r_[np.ones(len(Dtr)), np.zeros(len(Dtr))], C)
        a = ((Dva / sd) @ w[:-1] > 0).mean()
        if best is None or a > best[0]: best = (a, C, w)
    w = best[2]
    return float(((Dte / sd) @ w[:-1] > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", required=True, help="name=path.npz ...")
    ap.add_argument("--feature", default="last_instr", choices=["last_instr", "mean_instr", "mean_img", "last_tok"])
    ap.add_argument("--layers", default="", help="comma list; default all")
    ap.add_argument("--out", default=str(ROOT / "results/probes"))
    ap.add_argument("--no-controls", action="store_true", help="skip shuffled-label and diff probes (faster)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pairs, tab = load_meta()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}

    for spec in a.acts:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        X = z[a.feature].astype(np.float32)  # [N, L+1, d]
        y, split, group, pid = z["label"], z["split"], z["group"], z["pair_id"]
        N, nL, d = X.shape
        partner = pair_index(pid, y)
        meta = [tab[(p, int(l))] for p, l in zip(pid, y)]
        text = np.array([m["instruction"].lower() for m in meta])
        smash = np.array(["smash" in t for t in text])
        scene_only = np.array([m["changed"] == "scene" for m in meta])
        high = np.array([pairs[p]["safety_label_confidence"] == "high" for p in pid])
        tr, va, te = split == "train", split == "val", split == "test"
        subsets = {"all": te, "non_smash_pairs": te & ~smash & ~(smash[partner]),
                   "no_scene_only": te & ~scene_only, "scene_only": te & scene_only, "high_confidence": te & high}
        layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(nL))
        L = {}
        for l in layers:
            C, s_te, s_va = probe_layer(X[tr, l], y[tr], X[va, l], y[va], X[te, l])
            score = np.full(N, np.nan); score[te] = s_te
            pred = (score > 0).astype(int)
            ok = (pred == y)
            row = {"C": C, "val_auroc": float(auroc(s_va, y[va])), "test_acc": float(ok[te].mean()),
                   "test_auroc": float(auroc(s_te, y[te])), "test_acc_ci": boot_ci(ok[te], group[te], rng)}
            hm = te & (y == 1)
            row["test_pairwise_acc"] = float((score[hm] > score[partner[hm]]).mean())
            for k, m in subsets.items():
                if k != "all" and m.sum() > 0: row[f"test_acc[{k}]"] = float(ok[m].mean()); row[f"n[{k}]"] = int(m.sum())
            if not a.no_controls:
                _, s_sh, _ = probe_layer(X[tr, l], y[tr], X[va, l], y[va], X[te, l], shuffle_rng=rng)
                row["test_acc_shuffled_labels"] = float(((s_sh > 0).astype(int) == y[te]).mean())
                row["test_pairwise_acc_diff_probe"] = diff_probe_layer(X[:, l], y, pid, split, None, rng)
            L[l] = row
            print(f"{name} layer {l:2d}  acc {row['test_acc']:.3f} {tuple(round(c, 3) for c in row['test_acc_ci'])}  "
                  f"auroc {row['test_auroc']:.3f}  pair {row['test_pairwise_acc']:.3f}", flush=True)
        best = max(L, key=lambda k: L[k]["val_auroc"])
        res[name] = {"feature": a.feature, "best_layer_by_val": best, "layers": L,
                     "baseline_smash_keyword_test_acc": float(((smash[te]).astype(int) == y[te]).mean()),
                     "n_train_val_test": [int(tr.sum()), int(va.sum()), int(te.sum())]}
        print(f"== {name}: best layer by val = {best}, test acc {L[best]['test_acc']:.3f}, "
              f"smash-keyword baseline {res[name]['baseline_smash_keyword_test_acc']:.3f}")

    json.dump(res, open(out / f"{a.feature}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in res.items():
        ls = sorted(r["layers"]); acc = [r["layers"][l]["test_acc"] for l in ls]
        ci = np.array([r["layers"][l]["test_acc_ci"] for l in ls])
        ax.plot(ls, acc, label=name); ax.fill_between(ls, ci[:, 0], ci[:, 1], alpha=0.15)
        if not a.no_controls:
            ax.plot(ls, [r["layers"][l]["test_acc_shuffled_labels"] for l in ls], ":", color="gray", lw=0.8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8); ax.axhline(next(iter(res.values()))["baseline_smash_keyword_test_acc"], color="red", ls="--", lw=0.8, label="smash-keyword rule")
    ax.set_xlabel("layer"); ax.set_ylabel("test accuracy"); ax.set_title(f"Linear probe, feature={a.feature}"); ax.legend(); fig.tight_layout()
    fig.savefig(out / f"{a.feature}_layers.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
