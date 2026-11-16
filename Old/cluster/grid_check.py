#!/usr/bin/env python3
"""Check the grid scenes before spending GPU time on them. No model needed.

For every pair in semantic_safety_pairs/grid_pairs.jsonl: load the scene, let it settle with the arm still, and report
  - scenes that fail to load,
  - objects that are missing, off the table, or that slid sideways while settling (spawned overlapping something),
  - outcome checks that cannot be evaluated (e.g. they name an object the scene lacks),
  - outcome checks that are already true before the robot moves (would count as "carried out" for free).
Saves one upright picture per scene and contact sheets (sheet_<scenario>.jpg) to look at.

  python cluster/grid_check.py --out $VLA_HARM_ROOT/grid_check
"""
import argparse, json, os, gc, numpy as np
from PIL import Image, ImageDraw
from common import SSP, BENCH_BDDL, make_env, view, positions, load_jsonl, outcome, start_from_saved, saved_joints

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--scenario", default=""); ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--pairs", default="grid_pairs.jsonl", help="pair file in semantic_safety_pairs/, e.g. swap_pairs.jsonl")
ap.add_argument("--no_saved_starts", action="store_true", help="scenario A: use a fresh reset instead of LIBERO's saved start state (for machines without torch)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
INIT = os.path.join(os.path.dirname(BENCH_BDDL), "init_files")
rows = [r for r in load_jsonl(os.path.join(SSP, a.pairs)) if not a.scenario or r["scenario"] in a.scenario]
report, shots = {}, {}
for n, r in enumerate(rows):
    pid, prob = r["pair_id"], []
    try:
        joints = saved_joints(os.path.join(BENCH_BDDL, r["source_scene"]), os.path.join(INIT, r["source_init"]), 0) if r.get("source_init") else None
        env = make_env(os.path.join(BENCH_BDDL if r["root"] == "bench" else SSP, r["bddl"]))
        if r["init"] and not a.no_saved_starts:
            import torch
            env.reset(); obs = env.set_init_state(torch.load(os.path.join(INIT, r["init"]), weights_only=False)[0])
        elif r.get("source_init"):   # a benchmark scene with objects added: start it from the benchmark's saved start state
            obs = start_from_saved(env, joints, steps=1)
        else: obs = env.reset()
        insts = r.get("instances") or [b[:-5] for b in env.env.sim.model.body_names if b.endswith("_main")]
        p0 = positions(env, insts)
        for _ in range(a.steps): obs, *_ = env.step([0.]*6 + [-1.])
        p1 = positions(env, insts)
        prob += [f"missing {k}" for k in insts if k not in p1]
        slid = {k: 100 * float(np.linalg.norm((p1[k] - p0[k])[:2])) for k in p0 if k in p1}   # sideways only: everything drops a few cm onto the table
        prob += [f"{k} slid {v:.0f} cm while settling" for k, v in slid.items() if v > 4]
        prob += [f"{k} is off the table" for k, v in p1.items() if abs(v[0]) > 0.6 or abs(v[1]) > 0.6 or v[2] < 0.3]
        for cond, c in r["checks"].items():
            v = outcome(env, c, p1)
            if v is not False: prob.append(f"check for the {cond} instruction: {'already true before the robot moves' if v is True else v}")
        img = Image.fromarray(view(obs)); img.save(f"{a.out}/{pid}.png"); shots[pid] = (r["scenario"], img)
        env.close(); del env
    except Exception as e:
        prob.append(f"FAILED TO LOAD: {type(e).__name__}: {str(e).splitlines()[-1][:200] if str(e) else ''}")
    if prob: report[pid] = prob
    print(f"[{n+1}/{len(rows)}] {pid}: {'; '.join(prob) if prob else 'ok'}", flush=True)
    gc.collect()
json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
for sc in sorted({s for s, _ in shots.values()}):
    ims = [(k, im) for k, (s, im) in shots.items() if s == sc]; cols = 6; W = 256
    sheet = Image.new("RGB", (cols * W, ((len(ims) + cols - 1) // cols) * (W + 14)), "white"); d = ImageDraw.Draw(sheet)
    for i, (k, im) in enumerate(ims):
        x, y = (i % cols) * W, (i // cols) * (W + 14); sheet.paste(im.resize((W, W)), (x, y + 14)); d.text((x + 2, y + 1), k[:40], fill="black")
    sheet.save(f"{a.out}/sheet_{sc}.jpg", quality=85)
print(f"\nGRID CHECK: {len(rows) - len(report)}/{len(rows)} scenes clean, {len(report)} with notes (report.json, sheet_*.jpg in {a.out})")
