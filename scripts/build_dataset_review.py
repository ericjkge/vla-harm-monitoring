"""Build results/dataset_review/REPORT.md (+ example images) from minimal_pairs_500.jsonl."""
import json, shutil, collections as C, random
from pathlib import Path
R = [json.loads(l) for l in open("libero_safety/minimal_pairs_500.jsonl")]
T = {(x["pair_id"], x["label"]): x for x in map(json.loads, open("results/dataset_review/probe_table.jsonl"))}
rd = json.load(open("results/dataset_review/probe_readiness.json"))
out = Path("results/dataset_review"); (out / "images").mkdir(exist_ok=True)
rng = random.Random(0)
def tab(rows, head): return "\n" + "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
def cnt(key): c = C.Counter(key(r) for r in R); return sorted(c.items(), key=lambda kv: -kv[1])
def img(r):
    ps = []
    for lab, side in ((0, "harmless"), (1, "harmful")):
        src = T[(r["pair_id"], lab)]["images"]["agentview"]; dst = out / "images" / f"{r['pair_id']}_{side}.png"
        shutil.copy(src, dst); ps.append(f"![{side}](images/{dst.name})")
    return " ".join(ps)
def ex(r, show_img=False):
    h, n = r["harmless"], r["harmful"]
    s = (f"- **{r['pair_id']}** ({r['split']}, {r['hazard_family']}, {r['harm_type']}, {r['semantic_level']}; changed: {r['changed_component']})  \n"
         f"  harmless: “{h['instruction']}”  \n  harmful: “{n['instruction']}”  \n  scene: `{h['scene']['scene_id'].split('/')[-1][:70]}`"
         + (f" → `{n['scene']['scene_id'].split('/')[-1][:70]}`" if n['scene']['scene_id'] != h['scene']['scene_id'] or n['scene']['additions'] != h['scene']['additions'] else " (same scene)")
         + f"  \n  why harmful: {r['harm_mechanism']}")
    return s + ("\n\n  " + img(r) if show_img else "")
def pick(f, k=3, seen=set()):
    pool = [r for r in R if f(r)]; rng.shuffle(pool); res = []
    fams = set()
    for r in pool:
        if r["hazard_family"] in fams and len(fams) < 3: continue
        fams.add(r["hazard_family"]); res.append(r)
        if len(res) == k: break
    return res
L = [
 "Source: `libero_safety/minimal_pairs_500.jsonl`. Each pair = one harmless and one harmful sample, differing in exactly one component (verb, object or scene). Every pair is still `pending_review`: the labels come from the hand-written lexicon, not a human check, so **please flag wrong labels**.", "",
 "## 1. Probe readiness", "",
 f"Checked by `scripts/check_probe_readiness.py`. Flattened to **{rd['rows']} samples** (500 harmless + 500 harmful) in `probe_table.jsonl`.", "",
 "**Passes:** exactly 500 pairs and 1000 labelled samples; all labels consistent; split 300/100/100 pairs, perfectly balanced 0/1 in every split; no `group` spans two splits; every sample's scene resolves to rendered agentview/wrist/front images (none missing, 0 failed scenes, `scene_ok` true everywhere) and to lighting/camera variants; all components are ID (inside LIBERO-Safety).", "",
 "**Caveats that will affect probe results:**", "",
 f"1. **Text-only is ambiguous for scene pairs.** {rd['scene_only_pairs_text_identical']} pairs (8%) change only the scene, so the instruction is identical across labels (`{'`, `'.join(rd['texts_with_both_labels'])}`). A text-only probe cannot get these right, so they need image-conditioned activations. Consider reporting results with and without them.",
 "2. **Lexical shortcut: “smash”.** 262/500 pairs (52%) are verb swaps to “smash”. A rule “label harmful iff text contains *smash*” scores 80% on train, 72% on val and 68% on test. A probe must clearly beat this; also report accuracy on the non-smash subset.",
 "3. **Low text diversity.** Only 251 unique instruction strings (170 harmless, 83 harmful) among 1000 samples; same string appears many times with different scenes. Train and test share a few strings (train∩test 6, train∩val 12). Effective sample size is closer to ~57 groups than 1000 rows, so use group-wise CV and expect wide confidence intervals.",
 "4. **Hazard families differ across splits** (groups never span splits): e.g. `pour_onto` is entirely in test, `hand_tools`, `stove_verb`, `micro_scene` entirely in train, `candle_obj`/`push_off` mostly val/test. This makes test an out-of-family generalisation test, which is stricter than a random split; a drop vs val is expected rather than a bug.",
 "5. **Label confidence:** 109 pairs have `medium` safety-label confidence and 166 are `uncertain` in the validator (`validation_results_500.jsonl`), mainly stove turn-on/off and microwave-close cases. Consider also reporting results on the `high`-confidence subset.",
 "6. **Scene-only pairs are confounded**: 17 of them also change the layout, and the `turn off` cases are near no-ops if the stove is not shown as on (see `confounds`).",
 "7. Raw pair JSON still says `image: available false`; images are linked through `scene_manifest.json` (key = `render_scenes.scene_key`), already done in `probe_table.jsonl`. No sklearn/torch in the base Python here; extraction should run in the appropriate env.", "",
 "**Verdict:** a linear probe can be run on this data as is (labels, splits, images all present). Interpret results against the smash-keyword and text-only baselines.", "",
 "## 2. Composition", "",
 "**By changed component**", tab(cnt(lambda r: r["changed_component"]), ["changed", "pairs"]), "",
 "**By harm type** (explicit = harm stated in the words; implied = needs commonsense)", tab(cnt(lambda r: r["harm_type"]), ["harm_type", "pairs"]), "",
 "**By semantic level** (L0 explicit harm, L1 physical commonsense violation, L2 contextual trap)", tab(cnt(lambda r: r["semantic_level"]), ["level", "pairs"]), "",
 "**By safety category**", tab(cnt(lambda r: r["safety_category"]), ["category", "pairs"]), "",
 "**By split × changed component**", tab(sorted((f"{a} / {b}", n) for (a, b), n in C.Counter((r["split"], r["changed_component"]) for r in R).items()), ["split / changed", "pairs"]), "",
 "**By hazard family**", tab(cnt(lambda r: r["hazard_family"]), ["family", "pairs"]), "",
 "**Origin**: 117 `original` pairs (earlier reviewed 129 minus dropped) and 383 `expansion` pairs.", "",
 "## 3. Examples", ""]
secs = [("Verb swap, explicit harm (L0)", lambda r: r["changed_component"] == "verb" and r["harm_type"] == "explicit_harm"),
 ("Verb swap, implied harm (L1)", lambda r: r["changed_component"] == "verb" and r["harm_type"] == "implied_harm"),
 ("Object swap, explicit harm", lambda r: r["changed_component"] == "object" and r["harm_type"] == "explicit_harm"),
 ("Object swap, implied harm", lambda r: r["changed_component"] == "object" and r["harm_type"] == "implied_harm"),
 ("Scene swap (identical text, only the scene differs)", lambda r: r["changed_component"] == "scene"),
 ("Semantic level L2 (contextual trap; only 3 pairs)", lambda r: r["semantic_level"] == "L2")]
for t, f in secs:
    L += [f"### {t}", ""]
    for i, r in enumerate(pick(f, 3 if "L2" not in t else 3)): L += [ex(r, show_img=(i == 0)), ""]
L += ["### By hazard family (one each)", ""]
for fam, _ in cnt(lambda r: r["hazard_family"]):
    r = next(r for r in R if r["hazard_family"] == fam); L += [ex(r), ""]
L += ["## 4. Questions for reviewers", "",
 "- Are the harmful labels right (especially `medium` confidence and stove/microwave cases)?",
 "- Is 52% “smash” acceptable, or should we cap it / report non-smash separately?",
 "- Should scene-only pairs (identical text) stay in the main probe set?",
 "- Are the family-disjoint splits what we want for the headline number?"]
(out / "REPORT.md").write_text("\n".join(L))
