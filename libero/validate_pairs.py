"""Rule-based validation of minimal pairs (no LLM judge): structure, minimality, distribution metadata, safety labels, duplicates.

    python data_generation/validate_pairs.py [--pairs minimal_pairs_pilot.jsonl] [--report dataset_report.md]

Per pair status: invalid (a hard rule fails), uncertain (hard rules pass but evidence is not strong enough to count it as verified:
distribution confidence != high, safety confidence != high, or the lexicon cannot decide a label), valid (everything passes at high confidence).
Near-duplicates are flagged but do not change status.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from lexicon import OBJECTS, expected_label, VERBS

HERE = Path(__file__).parent
DIST = json.load(open(HERE / "libero_distribution.json"))
COND = {"id_verb": ("verb", "both_ID"), "ood_verb": ("verb", "any_OOD"), "id_object": ("object", "both_ID"),
        "ood_object": ("object", "any_OOD"), "id_scene": ("scene", "both_ID"), "ood_scene": ("scene", "any_OOD")}
ORDER = list(COND)


def status(comp, s):
    if comp == "verb":
        return "OOD" if any(DIST["verbs"].get(v, {"status": "OOD"})["status"] == "OOD" for v in s["verb"].split(" and ")) else "ID"
    if comp == "object": return DIST["objects"].get(s["object"], {"status": "OOD"})["status"]
    return DIST["scenes"][s["scene"]["layout"]]["status"]


def scene_key(sc): return (sc["layout"], sc["asset_id"], tuple(sorted((a["object"], a["placement"]) for a in sc["added_objects"])))
def comp_key(comp, s): return scene_key(s["scene"]) if comp == "scene" else s[comp]


def validate(p):
    inv, unc, warn = [], [], []
    H, F = p["harmless"], p["harmful"]
    comp, rule = COND.get(p["condition"], (None, None))
    if comp is None: return ["unknown condition"], [], []
    # 1/2 exactly one component changes, everything else (incl. relation/target) identical
    differing = [c for c in ("verb", "object", "scene") if comp_key(c, H) != comp_key(c, F)]
    if differing != [comp]: inv.append(f"components that differ: {differing}, expected only [{comp}]")
    for c in ("relation", "target"):
        if H[c] != F[c]: inv.append(f"frame field '{c}' differs")
    if p["changed_component"] != comp: inv.append("changed_component disagrees with condition")
    if comp != "scene" and H["instruction"] == F["instruction"]: inv.append("identical instruction but component is not scene")
    if comp == "scene" and H["instruction"] != F["instruction"]: inv.append("scene condition must keep the instruction text identical")
    # instruction text consistent with structured fields
    for role, s in (("harmless", H), ("harmful", F)):
        t = s["instruction"].lower()
        vs = s["verb"].split(" and ")
        if not t.startswith(vs[0]): inv.append(f"{role}: instruction does not start with verb '{vs[0]}'")
        if s["object"] in OBJECTS and s["object"] != "flat_stove" and OBJECTS[s["object"]]["phrase"] not in t: inv.append(f"{role}: object phrase missing from instruction")
    # 4 labels
    if (H["safety_label"], F["safety_label"]) != ("harmless", "harmful"): inv.append("safety labels are not (harmless, harmful)")
    # 3 condition vs distribution, recomputed from libero_distribution.json (stored metadata must agree)
    sh, sf = status(comp, H), status(comp, F)
    if p["distribution"].get(f"harmless_{comp}") != sh or p["distribution"].get(f"harmful_{comp}") != sf: inv.append("stored distribution metadata disagrees with libero_distribution.json")
    if rule == "both_ID" and not (sh == "ID" and sf == "ID"): inv.append(f"condition needs both {comp}s ID, got {sh}/{sf}")
    if rule == "any_OOD" and "OOD" not in (sh, sf): inv.append(f"condition needs at least one OOD {comp}, got {sh}/{sf}")
    # 5 unchanged components verified ID
    for c in ("verb", "object", "scene"):
        if c != comp and (status(c, H) != "ID" or status(c, F) != "ID"): inv.append(f"unchanged component '{c}' is not ID")
    for role, s in (("harmless", H), ("harmful", F)):
        if s["target"] and DIST["objects"].get(s["target"], {"status": "OOD"})["status"] != "ID": inv.append(f"{role}: target not ID")
    # 7 harm evidence: independent lexicon rule
    fam = p.get("hazard_family")
    for role, s in (("harmless", H), ("harmful", F)):
        exp = expected_label(fam, s)
        if exp is None: unc.append(f"{role}: lexicon cannot confirm the label")
        elif exp != s["safety_label"]: inv.append(f"{role}: lexicon says {exp}, pair says {s['safety_label']}")
    if not p.get("harmfulness_explanation"): unc.append("no harmfulness explanation")
    if p.get("safety_label_confidence") != "high": unc.append(f"safety label confidence {p.get('safety_label_confidence')}")
    # 8 uncertain distribution is not counted as verified
    for k, v in p["distribution_confidence"].items():
        if v != "high": unc.append(f"distribution confidence for {k} is {v}")
    for a in F["scene"]["added_objects"] + H["scene"]["added_objects"]:
        if DIST["objects"].get(a["object"], {"status": "OOD"})["status"] == "OOD" and comp == "scene" and p["condition"] == "id_scene": inv.append("id_scene pair adds an OOD object")
    if F["scene"]["config"] == "constructed" or H["scene"]["config"] == "constructed": warn.append("constructed scene configuration (image must be built)")
    return inv, unc, warn


def near_duplicates(pairs):
    flags = defaultdict(list)
    seen = {}
    for p in pairs:
        k = (p["harmless"]["instruction"], p["harmful"]["instruction"], scene_key(p["harmless"]["scene"]), scene_key(p["harmful"]["scene"]))
        if k in seen: flags[p["pair_id"]].append(f"exact duplicate of {seen[k]}")
        seen[k] = seen.get(k, p["pair_id"])
    by_instr = defaultdict(list)
    for p in pairs:
        if p["changed_component"] != "scene": by_instr[(p["harmless"]["instruction"], p["harmful"]["instruction"])].append(p["pair_id"])
    for ids in by_instr.values():
        for i in ids[1:]:
            if i not in flags: flags[i].append(f"same instruction pair as {ids[0]} (different scene)")
    by_harm = defaultdict(list)
    for p in pairs: by_harm[(p["condition"], p["harmful"]["instruction"], scene_key(p["harmful"]["scene"]))].append(p["pair_id"])
    for ids in by_harm.values():
        for i in ids[1:]:
            if i not in flags: flags[i].append(f"same harmful instruction+layout as {ids[0]} (different harmless side)")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "minimal_pairs_pilot.jsonl"))
    ap.add_argument("--out", default=str(HERE / "validation_results.jsonl"))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs)]
    dups = near_duplicates(pairs)
    res = []
    for p in pairs:
        inv, unc, warn = validate(p)
        st = "invalid" if inv else ("uncertain" if unc else "valid")
        exact = [d for d in dups.get(p["pair_id"], []) if d.startswith("exact")]
        if exact: st = "invalid"; inv.append(exact[0])
        res.append({"pair_id": p["pair_id"], "condition": p["condition"], "status": st, "errors": inv, "needs_review": unc, "warnings": warn, "near_duplicate_flags": dups.get(p["pair_id"], [])})
    Path(a.out).write_text("\n".join(json.dumps(r) for r in res) + "\n")
    tab = {c: Counter() for c in ORDER}
    for r in res: tab[r["condition"]][r["status"]] += 1
    print(f"{'condition':12} valid invalid uncertain total  near-dup flags")
    for c in ORDER:
        n = sum(tab[c].values()); nd = sum(1 for r in res if r["condition"] == c and r["near_duplicate_flags"])
        print(f"{c:12} {tab[c]['valid']:5} {tab[c]['invalid']:7} {tab[c]['uncertain']:9} {n:5}  {nd}")
    if a.report: write_report(a.report, pairs, res, tab)


def write_report(path, pairs, res, tab):
    from report_text import render
    Path(path).write_text(render(pairs, res, tab, DIST))


if __name__ == "__main__":
    main()
