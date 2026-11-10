#!/usr/bin/env python3
"""Render the settled first frame of every generated scene (semantic_safety_pairs/scenes/*.bddl).
No model needed. Writes frames/<pair_id>.png plus _warnings.json (objects that moved while settling)
and _failures.json. Resumable. Shard with --shard i --nshards n to run several jobs in parallel.

  python cluster/render_scenes.py --out $VLA_HARM_ROOT/frames_gen --shard 0 --nshards 4
"""
import argparse, json, os, gc, numpy as np
from common import SSP, make_env, settle, frame, view, positions, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--index", default=os.path.join(SSP, "scene_index.jsonl"))
a = ap.parse_args(); os.makedirs(f"{a.out}/_view", exist_ok=True)
rows = load_jsonl(a.index)[a.shard::a.nshards]
warn_p, fail_p = f"{a.out}/_warnings_{a.shard}.json", f"{a.out}/_failures_{a.shard}.json"
warnings = json.load(open(warn_p)) if os.path.exists(warn_p) else {}
failures = {}
todo = [r for r in rows if not os.path.exists(f"{a.out}/{r['pair_id']}.png")]
print(f"shard {a.shard}/{a.nshards}: {len(rows)} scenes, {len(todo)} to do", flush=True)
for i, r in enumerate(todo):
    try:
        env = make_env(os.path.join(SSP, r["bddl"]))
        obs = env.reset(); p0 = positions(env, r["instances"])
        for _ in range(30): obs, *_ = env.step([0.]*6 + [-1.])
        p1 = positions(env, r["instances"])
        moved = {k: float(np.linalg.norm(p1[k]-p0[k])) for k in p0 if k in p1}
        if any(v > 0.08 for v in moved.values()) or len(p1) < len(r["instances"]):
            warnings[r["pair_id"]] = {"moved_cm": {k: round(v*100, 1) for k, v in moved.items() if v > 0.03},
                                     "missing": [k for k in r["instances"] if k not in p1]}
            json.dump(warnings, open(warn_p, "w"), indent=1)
        # <id>.png is the model input (rotated and cropped like OpenVLA's eval); _view/<id>.png is upright, for people
        from PIL import Image
        Image.fromarray(view(obs)).save(f"{a.out}/_view/{r['pair_id']}.png")
        frame(obs).save(f"{a.out}/{r['pair_id']}.png"); env.close(); del env
    except Exception as e:
        failures[r["pair_id"]] = str(e).splitlines()[-1][:200]; json.dump(failures, open(fail_p, "w"), indent=1)
    gc.collect()
    if i % 10 == 0: print(f"{i}/{len(todo)} warnings={len(warnings)} failures={len(failures)}", flush=True)
print("done", flush=True)
