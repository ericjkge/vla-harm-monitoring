#!/usr/bin/env python3
"""Closed-loop rollouts: does the VLA actually execute the harmful instruction?

For each pair, runs the model in the simulator for --steps steps on the safe and the unsafe
instruction (same scene, same seed), saves an mp4 per rollout, the action sequence, the
first-step hidden states, and a per-step distance from the gripper to each object of interest,
so "did it move toward the hand / the stove" can be scored without watching every video.

  # the 15 LIBERO-Safety semantic tasks + our safe twins (needs a GPU; ~1 h on an A100 at 200 steps)
  python cluster/rollouts.py --set benchmark --out $VLA_HARM_ROOT/rollouts_bench --steps 200
  # a subset of generated pairs (scenes from semantic_safety_pairs/scenes)
  python cluster/rollouts.py --set generated --limit 60 --out $VLA_HARM_ROOT/rollouts_gen --steps 200
"""
import argparse, json, os, pickle, numpy as np, imageio
from common import SSP, BENCH_BDDL, make_env, settle, frame, positions, load_vla, predict_action, hidden_states, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["benchmark", "generated"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-10")
ap.add_argument("--unnorm_key", default="libero_10"); ap.add_argument("--four_bit", action="store_true")
a = ap.parse_args(); os.makedirs(f"{a.out}/videos", exist_ok=True)

if a.set == "benchmark":
    pairs = [dict(pair_id=r["task_id"], bddl=os.path.join(BENCH_BDDL, r["bddl"]), safe=r["safe_instruction"],
                  unsafe=r["unsafe_instruction"], tier=r["tier"], category=r["category"], instances=None)
             for r in load_jsonl(os.path.join(SSP, "libero_ssr_matched_pairs.jsonl"))]
else:
    pairs = [dict(pair_id=r["pair_id"], bddl=os.path.join(SSP, r["bddl"]), safe=r["safe_instruction"], unsafe=r["unsafe_instruction"],
                  tier=r["tier"], category=r["category"], instances=r["instances"]) for r in load_jsonl(os.path.join(SSP, "scene_index.jsonl"))]
if a.limit: pairs = pairs[:a.limit]

vla, proc, dtype = load_vla(a.model, a.four_bit)
rec_p = f"{a.out}/records.pkl"
records = pickle.load(open(rec_p, "rb")) if os.path.exists(rec_p) else []
done = {(r["pair_id"], r["cond"]) for r in records}
print(f"{len(pairs)} pairs, {len(done)} rollouts already done", flush=True)

def gripper_pos(env): return np.array(env.env.sim.data.site_xpos[env.env.sim.model.site_name2id("gripper0_grip_site")])

for p in pairs:
    for cond in ("safe", "unsafe"):
        if (p["pair_id"], cond) in done: continue
        instr = p[cond]
        env = make_env(p["bddl"]); obs = settle(env)
        insts = p["instances"] or [b[:-5] for b in env.env.sim.model.body_names if b.endswith("_main")]
        obj0 = positions(env, insts)
        img0 = frame(obs); hs = hidden_states(vla, proc, dtype, img0, instr)
        vid, acts, dists = [np.array(img0)], [], []
        for t in range(a.steps):
            act = predict_action(vla, proc, dtype, frame(obs), instr, a.unnorm_key)
            acts.append(act); obs, _, done_flag, _ = env.step(act.tolist())
            g = gripper_pos(env); cur = positions(env, insts)
            dists.append({k: float(np.linalg.norm(g - v)) for k, v in cur.items()})
            vid.append(obs["agentview_image"][::-1])
            if done_flag: break
        env.close()
        path = f"{a.out}/videos/{p['pair_id']}_{cond}.mp4"; imageio.mimsave(path, vid, fps=20)
        records.append(dict(pair_id=p["pair_id"], cond=cond, instr=instr, tier=p["tier"], category=p["category"], hs=hs,
                            actions=np.array(acts), gripper_obj_dist=dists, obj_start=obj0, video=path))
        pickle.dump(records, open(rec_p, "wb"))
        print(f"{p['pair_id']} {cond} done ({len(acts)} steps) -> {path}", flush=True)
print("all done:", rec_p)
