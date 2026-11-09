#!/usr/bin/env python3
"""
Assign generated safe/unsafe pairs (pairs.jsonl) to the 15 LIBERO-Safety semantic scenes
(ssr_scene_objects.json) so each instruction can be run against an already-rendered frame.

grounding = "strict":        every object phrase mentioned in either instruction is in the scene
grounding = "targets_only":  only the pair's target_objects are in the scene (instruction may
                             mention an absent object such as a basket; weaker visual match)

Also derives a template key (instruction with object phrases replaced by {obj}) for
leave-one-template-out evaluation, and a same_verb flag for the verb-controlled subset.

Usage: python3 match_pairs_to_scenes.py            -> writes ssr_scene_pairs.jsonl
"""
import json, re, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_pairs import NAME

NAME = dict(NAME)
NAME.update({"water_ball": "water", "white_place_box": "storage box", "cap_1_with_hand": "cap in the person's hand"})

def base(t):
    """Normalize a sim object type to a base category."""
    t = t.replace("chefmate_8_frypan", "frypan")
    t = re.sub(r"_with_hand$", "", t)
    t = re.sub(r"(__|_)\d+$", "", t)
    return t

SCENES = json.load(open("ssr_scene_objects.json"))
scene_base = {k: {base(o) for o in v["objects"]} for k, v in SCENES.items()}
scene_hand = {k: [o for o in v["objects"] if o.endswith("_with_hand") or o == "left_hand"] for k, v in SCENES.items()}
for k in scene_base:
    scene_base[k] |= {"table"}                      # a table is in every scene
    if {"white_cabinet", "wooden_cabinet"} & scene_base[k]: scene_base[k].add("cabinet")
    if "flat_stove" in scene_base[k]: scene_base[k].add("stove")

# phrase -> sim type, longest phrase first so "porcelain mug" wins over "mug"
PHRASES = sorted(((v, k) for k, v in NAME.items() if "hand" not in v), key=lambda x: -len(x[0]))
GENERIC = {"cabinet": "cabinet", "stove": "stove", "table": "table", "cutting board": "cutting_board", "basket": "basket",
           "bowl": "bowl", "mug": "mug", "plate": "plate"}

HAND_RE = re.compile(r"hand holding the ([a-z ]+?)(?:\s|$|,)|person's hand|my hand|hand on the|my palm|my fingers| me\b| for me|hand me|hand the|to the person|at the person|holding the")

def mentions(text):
    """Return (set of base types mentioned, templated text, hand requirement)."""
    t = " " + text.lower() + " "
    found = set()
    for phrase, sim in PHRASES:
        pat = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pat, t):
            found.add(base(sim)); t = re.sub(pat, "{obj}", t)
    for phrase, sim in GENERIC.items():
        pat = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pat, t):
            found.add(sim); t = re.sub(pat, "{obj}", t)
    hand = None
    m = re.search(r"hand holding the \{obj\}|hand holding the ([a-z ]+)", text.lower())
    if "hand" in text.lower() or " me" in text.lower() or "person" in text.lower():
        hand = "any"
        m2 = re.search(r"hand holding the ([a-z ]+?)(?: with| and|$)", text.lower())
        if m2: hand = m2.group(1).strip()
    return found, re.sub(r"\s+", " ", t).strip(), hand

def hand_ok(scene_id, req):
    if req is None: return True
    hands = scene_hand[scene_id]
    if not hands: return False
    if req == "any": return True
    # specific held object, e.g. "soda can" -> can_of_soda_3_with_hand
    key = req.replace("soda can", "can_of_soda").replace(" ", "_")
    return any(key.split("_")[0] in h for h in hands)

pairs = [json.loads(l) for l in open("pairs.jsonl")]
rows, stats = [], collections.Counter()
for p in pairs:
    ms, ts, hs = mentions(p["safe_instruction"])
    mu, tu, hu = mentions(p["unsafe_instruction"])
    mentioned = (ms | mu) - {"bowl", "mug", "plate"} if False else (ms | mu)
    targets = {base(o) for o in p["target_objects"] if not (o.endswith("_with_hand") or o == "left_hand")}
    need_hand = hs or hu or None
    if need_hand is None and any(o.endswith("_with_hand") or o == "left_hand" for o in p["target_objects"]):
        need_hand = "any"
    for sid, sobjs in scene_base.items():
        # generic words: "bowl" satisfied by any *_bowl in scene, etc.
        def present(x): return x in sobjs or any(o.endswith("_" + x) or o == x for o in sobjs)
        if not hand_ok(sid, need_hand): continue
        if all(present(x) for x in mentioned): grounding = "strict"
        elif all(present(x) for x in targets): grounding = "targets_only"
        else: continue
        vs, vu = p["safe_instruction"].split()[0], p["unsafe_instruction"].split()[0]
        rows.append(dict(task_id=sid, pair_id=p["pair_id"], category=p["category"], tier=p["tier"],
                         grounding=grounding, safe_instruction=p["safe_instruction"], unsafe_instruction=p["unsafe_instruction"],
                         template_safe=ts, template_unsafe=tu, verb_safe=vs, verb_unsafe=vu, same_verb=(vs == vu)))
        stats[grounding] += 1

with open("ssr_scene_pairs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")

print("pairs assigned:", len(rows), dict(stats))
print("unique generated pairs used:", len({r['pair_id'] for r in rows}))
print("same-verb pairs (strict):", sum(1 for r in rows if r["same_verb"] and r["grounding"] == "strict"),
      "| same-verb (any):", sum(1 for r in rows if r["same_verb"]))
print("unique unsafe templates:", len({r['template_unsafe'] for r in rows}))
print("\nper scene (strict / targets_only):")
for sid in SCENES:
    s = sum(1 for r in rows if r["task_id"] == sid and r["grounding"] == "strict")
    t = sum(1 for r in rows if r["task_id"] == sid and r["grounding"] == "targets_only")
    print(f"  {sid}: {s:3d} / {t:3d}")
print("\nby category (strict):", dict(collections.Counter(r["category"] for r in rows if r["grounding"] == "strict")))
