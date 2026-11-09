#!/usr/bin/env python3
"""
Write one LIBERO BDDL scene file per generated pair (pairs.jsonl) so every pair can be rendered
and its instructions run against an image that actually contains the objects they name.

Layout: large fixtures (cabinets, microwave, stove, basket, caddy, wine rack, keyboard) go to
side slots at the coordinates the LIBERO-Safety authors use; everything else goes on a central
grid with 2 cm placement boxes, so each scene is deterministic up to a seeded shuffle.
All objects start on the table (nothing is pre-placed on the stove or in the microwave).

Usage: python3 make_scene_bddls.py   -> scenes/<pair_id>.bddl and scene_index.jsonl
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_pairs import NAME

PROBLEM = {"kitchen_table": "LIBERO_Kitchen_Tabletop_Manipulation",
           "living_room_table": "LIBERO_Living_Room_Tabletop_Manipulation",
           "study_table": "LIBERO_Study_Tabletop_Manipulation"}
# side slots for large objects: (x, y) centres taken from the benchmark's own files
BIG_SLOTS = [(0.00, -0.28), (0.00, 0.30), (-0.40, 0.20), (-0.40, -0.15), (0.25, -0.28)]
BIG = {"wooden_cabinet", "white_cabinet", "microwave", "flat_stove", "basket", "desk_caddy",
       "wine_rack", "white_place_box", "keyboard"}
# central grid for everything else
GRID = [(x, y) for x in (-0.18, -0.06, 0.06, 0.18) for y in (-0.15, -0.05, 0.05, 0.15)]
BOX = 0.01   # half-width of the placement box (2 cm box, as the authors use)

def region(name, table, cx, cy, half):
    return f"""      ({name}
          (:target {table})
          (:ranges (
              ({cx-half:.3f} {cy-half:.3f} {cx+half:.3f} {cy+half:.3f})
            )
          )
          (:yaw_rotation (
              (0 0)
            )
          )
      )"""

def write_bddl(path, table, objects, targets, language, seed):
    rng = random.Random(seed)
    big = [o for o in objects if o in BIG]; small = [o for o in objects if o not in BIG]
    assert len(big) <= len(BIG_SLOTS) and len(small) <= len(GRID), (big, small)
    grid = GRID[:]; rng.shuffle(grid)
    place = {o: BIG_SLOTS[i] for i, o in enumerate(big)}
    place.update({o: grid[i] for i, o in enumerate(small)})
    inst = {o: f"{o}_1" for o in objects}
    regs = "\n".join(region(f"{inst[o]}_region", table, *place[o], BOX if o not in BIG else 0.01) for o in objects)
    objs = "\n".join(f"    {inst[o]} - {o}" for o in objects)
    ooi = "\n".join(f"    {inst[o]}" for o in targets)
    init = "\n".join(f"    (On {inst[o]} {table}_{inst[o]}_region)" for o in objects)
    txt = f"""(define (problem {PROBLEM[table]})
  (:domain robosuite)
  (:language {language})
    (:regions
{regs}
    )

  (:fixtures
    {table} - {table}
  )

  (:objects
{objs}
  )

  (:obj_of_interest
{ooi}
  )

  (:init
{init}
  )

)
"""
    open(path, "w").write(txt)

# template key + verb, for leave-one-template-out and the same-verb subset
PHRASES = sorted(NAME.values(), key=len, reverse=True)
def template(text):
    t = " " + text.lower() + " "
    for ph in PHRASES:
        t = re.sub(r"\b" + re.escape(ph) + r"\b", "{obj}", t)
    for ph in ("cutting board", "cabinet", "stove", "basket", "table", "bowl", "mug", "plate"):
        t = re.sub(r"\b" + ph + r"\b", "{obj}", t)
    return re.sub(r"\s+", " ", t).strip()

if __name__ == "__main__":
    os.makedirs("scenes", exist_ok=True)
    pairs = [json.loads(l) for l in open("pairs.jsonl")]
    with open("scene_index.jsonl", "w") as idx:
        for i, p in enumerate(pairs):
            path = f"scenes/{p['pair_id']}.bddl"
            write_bddl(path, p["scene"], p["scene_objects"], p["target_objects"], p["safe_instruction"], seed=i)
            vs, vu = p["safe_instruction"].split()[0], p["unsafe_instruction"].split()[0]
            idx.write(json.dumps(dict(pair_id=p["pair_id"], bddl=path, scene=p["scene"], category=p["category"], tier=p["tier"],
                                      harm_target=p["harm_target"], safe_instruction=p["safe_instruction"],
                                      unsafe_instruction=p["unsafe_instruction"], template_safe=template(p["safe_instruction"]),
                                      template_unsafe=template(p["unsafe_instruction"]), verb_safe=vs, verb_unsafe=vu,
                                      same_verb=(vs == vu))) + "\n")
    print("wrote", len(pairs), "scene files to scenes/ and scene_index.jsonl")
