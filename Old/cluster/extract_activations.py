#!/usr/bin/env python3
"""Hidden states for every generated pair on its rendered frame (no simulator; needs frames from render_scenes.py).
  python cluster/extract_activations.py --frames $VLA_HARM_ROOT/frames_gen --out $VLA_HARM_ROOT/gen500_records.pkl
Also extracts the 15 benchmark pairs if --bench_frames (from render_benchmark.py or the Colab frames) is given.
"""
import argparse, os, pickle, json
from PIL import Image
from common import SSP, load_vla, hidden_states, predict_action, load_jsonl
ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-10"); ap.add_argument("--four_bit", action="store_true")
ap.add_argument("--skip_warned", action="store_true", help="skip scenes listed in frames/_warnings_*.json")
a = ap.parse_args()
warned = set()
for f in os.listdir(a.frames):
    if f.startswith("_warnings"): warned |= set(json.load(open(os.path.join(a.frames, f))))
vla, proc, dtype = load_vla(a.model, a.four_bit)
recs = pickle.load(open(a.out, "rb")) if os.path.exists(a.out) else []
done = {r["pair_id"] for r in recs}
rows = [r for r in load_jsonl(os.path.join(SSP, "scene_index.jsonl")) if r["pair_id"] not in done
        and os.path.exists(f"{a.frames}/{r['pair_id']}.png") and not (a.skip_warned and r["pair_id"] in warned)]
print(f"{len(rows)} pairs to extract ({len(warned)} flagged by the settle check{' and skipped' if a.skip_warned else ''})", flush=True)
for i, r in enumerate(rows):
    img = Image.open(f"{a.frames}/{r['pair_id']}.png").convert("RGB")
    recs.append(dict(r, hs_safe=hidden_states(vla, proc, dtype, img, r["safe_instruction"]),
                     hs_unsafe=hidden_states(vla, proc, dtype, img, r["unsafe_instruction"]),
                     a0_safe=predict_action(vla, proc, dtype, img, r["safe_instruction"]),
                     a0_unsafe=predict_action(vla, proc, dtype, img, r["unsafe_instruction"]), warned=r["pair_id"] in warned))
    if i % 25 == 0: pickle.dump(recs, open(a.out, "wb")); print(f"{i}/{len(rows)}", flush=True)
pickle.dump(recs, open(a.out, "wb")); print("saved", len(recs), "->", a.out)
