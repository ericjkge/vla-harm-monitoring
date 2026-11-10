#!/usr/bin/env python3
"""Closed-loop rollouts: does the VLA actually execute the harmful instruction?

For each pair, runs the model in the simulator for --steps steps on the safe and the unsafe
instruction (same scene, same seed), saves an mp4 per rollout, the action sequence, the
first-step hidden states, and a per-step distance from the gripper to each object of interest,
so "did it move toward the hand / the stove" can be scored without watching every video.

  # the 15 LIBERO-Safety semantic tasks + our safe twins (needs a GPU; ~45 min on an A100 at 300 steps)
  python cluster/rollouts.py --set benchmark --out $VLA_HARM_ROOT/rollouts_bench --steps 300
  # a subset of generated pairs (scenes from semantic_safety_pairs/scenes)
  python cluster/rollouts.py --set generated --limit 60 --out $VLA_HARM_ROOT/rollouts_gen --steps 200
  # pipeline check: standard LIBERO-10 tasks the model was fine-tuned on; published success is roughly half.
  # If this scores 0, the rollout pipeline is broken and no other rollout result means anything.
  python cluster/rollouts.py --set sanity --limit 6 --out $VLA_HARM_ROOT/rollouts_sanity --steps 520
"""
import argparse, json, os, pickle, numpy as np, imageio
from common import SSP, BENCH_BDDL, make_env, settle, frame, view, env_action, positions, load_vla, predict_action, hidden_states, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["benchmark", "generated", "sanity"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-10")
ap.add_argument("--unnorm_key", default="libero_10"); ap.add_argument("--four_bit", action="store_true")
a = ap.parse_args(); os.makedirs(f"{a.out}/videos", exist_ok=True)

if a.set == "sanity":
    import glob, re
    pairs = []
    for b in sorted(glob.glob(os.path.join(BENCH_BDDL, "libero_10", "*.bddl"))):
        lang = re.search(r"\(:language ([^)]*)\)", open(b).read()).group(1).strip()
        init = os.path.join(os.path.dirname(BENCH_BDDL), "init_files", "libero_10", os.path.basename(b)[:-5] + ".pruned_init")
        if not os.path.exists(init): continue  # the repo has stray duplicates ("... copy 2.bddl") with no initial state
        pairs.append(dict(pair_id=os.path.basename(b)[:-5][:60], bddl=b, task=lang, tier="sanity", category="libero_10", instances=None, init=init))
elif a.set == "benchmark":
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
records = [r for r in records if r.get("controller") == "stock_osc_pose"]  # redo anything from before the controller fix
done = {(r["pair_id"], r["cond"]) for r in records}
print(f"{len(pairs)} pairs, {len(done)} rollouts already done", flush=True)

def gripper_pos(env): return np.array(env.env.sim.data.site_xpos[env.env.sim.model.site_name2id("gripper0_grip_site")])

for p in pairs:
    for cond in (("task",) if a.set == "sanity" else ("safe", "unsafe")):
        if (p["pair_id"], cond) in done: continue
        instr = p[cond]
        env = make_env(p["bddl"])
        if p.get("init"):  # standard LIBERO tasks start from the benchmark's saved initial state
            import torch
            env.reset(); obs = env.set_init_state(torch.load(p["init"], weights_only=False)[0])
            for _ in range(10): obs, *_ = env.step([0.]*6 + [-1.])
        else: obs = settle(env)
        insts = p["instances"] or [b[:-5] for b in env.env.sim.model.body_names if b.endswith("_main")]
        obj0 = positions(env, insts)
        off = [k for k, v in obj0.items() if abs(v[0]) > 0.6 or abs(v[1]) > 0.6 or v[2] < 0.3]
        if off: print(f"  note: {p['pair_id']} starts with objects away from the table: {off}", flush=True)
        img0 = frame(obs); hs = hidden_states(vla, proc, dtype, img0, instr)
        vid, acts, dists, grip, goal_step = [view(obs)], [], [], [], None
        for t in range(a.steps):
            act = predict_action(vla, proc, dtype, frame(obs), instr, a.unnorm_key)
            acts.append(act); obs, _, done_flag, _ = env.step(env_action(act).tolist())
            g = gripper_pos(env); cur = positions(env, insts)
            dists.append({k: float(np.linalg.norm(g - v)) for k, v in cur.items()}); grip.append(g)
            vid.append(view(obs))
            # The done flag is only the scene file's goal predicate. Several benchmark scenes satisfy it at step 0,
            # so it is recorded (first step it fires) and the rollout always runs the full horizon.
            if done_flag and goal_step is None: goal_step = t
            if done_flag and a.set == "sanity": break  # here the predicate is the real task-success check
        obj1 = positions(env, insts)
        env.close()
        path = f"{a.out}/videos/{p['pair_id']}_{cond}.mp4"; imageio.mimsave(path, vid, fps=20)
        records.append(dict(pair_id=p["pair_id"], cond=cond, instr=instr, tier=p["tier"], category=p["category"], hs=hs,
                            actions=np.array(acts), gripper_obj_dist=dists, goal_step=goal_step, gripper_xyz=np.array(grip), obj_end=obj1, obj_start=obj0, off_table_at_start=off, controller="stock_osc_pose", video=path))
        pickle.dump(records, open(rec_p, "wb"))
        print(f"{p['pair_id']} {cond} done ({len(acts)} steps, goal predicate first true at {goal_step}) -> {path}", flush=True)
if a.set == "sanity":
    ok = sum(r["goal_step"] is not None for r in records)
    print(f"SANITY: {ok}/{len(records)} standard LIBERO-10 tasks succeeded (published rate is roughly half; 0 means the pipeline is broken)")
print("all done:", rec_p)
