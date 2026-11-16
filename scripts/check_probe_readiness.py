"""Scan libero_safety/minimal_pairs_500.jsonl for linear-probe readiness and write
results/dataset_review/probe_readiness.json + probe_table.jsonl (one row per sample: 1000 rows)."""
import json, sys, os, collections as C
sys.path.insert(0, "libero_safety")
from render_scenes import scene_key, scene_status
R = [json.loads(l) for l in open("libero_safety/minimal_pairs_500.jsonl")]
man = json.load(open("libero_safety/scene_manifest.json"))
rows, issues = [], C.Counter()
for r in R:
    for lab, side in ((0, "harmless"), (1, "harmful")):
        s = r[side]; k = scene_key(s["scene"]); e = man.get(k)
        img = None
        if e is None: issues["scene_not_in_manifest"] += 1
        else:
            ok, why = scene_status(e)
            if not ok: issues["scene_status_bad"] += 1
            img = e["images"]
            for p in img.values():
                if not os.path.exists(p): issues["image_file_missing"] += 1
            nv = len(e.get("variants", {}))
            if nv < 2: issues["fewer_than_2_variants"] += 1
        if s["safety_label"] != side: issues["label_mismatch"] += 1
        rows.append(dict(pair_id=r["pair_id"], split=r["split"], group=r["group"], label=lab, instruction=s["instruction"],
                         changed=r["changed_component"], harm_type=r["harm_type"], family=r["hazard_family"], scene_key=k, images=img,
                         variants={v: d["images"] for v, d in (e or {}).get("variants", {}).items()}))
if len(rows) != 1000: issues["row_count"] += 1
by = C.Counter((x["split"], x["label"]) for x in rows)
txt = C.defaultdict(set)
for x in rows: txt[x["instruction"]].add(x["label"])
res = dict(rows=len(rows), per_split_label=sorted((f"{a}/{b}", n) for (a, b), n in by.items()), issues=dict(issues),
           texts_with_both_labels=[t for t, v in txt.items() if len(v) > 1],
           scene_only_pairs_text_identical=sum(r["changed_component"] == "scene" for r in R),
           unique_texts=len(txt), unique_scenes=len({x["scene_key"] for x in rows}))
json.dump(res, open("results/dataset_review/probe_readiness.json", "w"), indent=1)
with open("results/dataset_review/probe_table.jsonl", "w") as f:
    for x in rows: f.write(json.dumps(x) + "\n")
print(json.dumps(res, indent=1))
