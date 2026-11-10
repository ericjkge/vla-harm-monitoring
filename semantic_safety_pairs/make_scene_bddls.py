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
# Layout, sized from the benchmark authors' own placements (table ~ x in [-0.45, 0.35], y in [-0.4, 0.4]).
# Large furniture/appliances take side slots; wide flat objects (boards, pans, keyboards, hands) take
# 20 cm-spaced slots; small objects (cans, fruit, mugs) fill 10 cm-spaced slots between them.
# Objects spawned closer than their footprint interpenetrate and MuJoCo flings them off the table.
SIDE_SLOTS = [(0.00, 0.34), (0.00, -0.34), (-0.30, 0.34), (-0.30, -0.34), (0.26, 0.34), (0.26, -0.34)]
STOVE_SLOT = (-0.41, 0.21)
BIG = {"wooden_cabinet", "white_cabinet", "microwave", "flat_stove", "basket", "desk_caddy",
       "wine_rack", "white_place_box", "keyboard"}
WIDE = {"cutting_board__1", "frypan", "chefmate_8_frypan", "game_console", "plate", "salad_porcelain_plate",
        "notebook", "black_book", "dish_towel", "hammer", "knife", "scissors", "fork", "toy_train",
        "akita_black_bowl", "wood_bowl", "porcelain_bowl", "glazed_rim_porcelain_ramekin", "billiard_balls",
        "left_hand", "plate_with_hand", "can_of_soda_3_with_hand", "banana_4_with_hand", "apple_1_with_hand",
        "bottle_of_water_2_with_hand", "croissant_1_with_hand", "cap_1_with_hand", "bottle_of_alcohol_1_with_hand"}
WIDE_SLOTS = [(x, y) for x in (0.18, -0.02, -0.20) for y in (-0.18, 0.00, 0.18)]
SMALL_SLOTS = [(x, y) for x in (0.18, -0.02, -0.20) for y in (-0.09, 0.09)]
# Articulated furniture/appliances go under (:fixtures), as in the benchmark's own files; a stove
# declared under (:objects) fails to compile ("duplicate element flat_stove_1_main").
FIXTURE = {"flat_stove", "microwave", "white_cabinet", "wooden_cabinet", "wine_rack"}

def region(name, table, cx, cy, half, yaw=0.0):
    return f"""      ({name}
          (:target {table})
          (:ranges (
              ({cx-half:.3f} {cy-half:.3f} {cx+half:.3f} {cy+half:.3f})
            )
          )
          (:yaw_rotation (
              ({yaw} {yaw})
            )
          )
      )"""

def write_bddl(path, table, objects, targets, language, seed):
    rng = random.Random(seed)
    big = [o for o in objects if o in BIG]; wide = [o for o in objects if o in WIDE]
    small = [o for o in objects if o not in BIG and o not in WIDE]
    place = {}
    side = SIDE_SLOTS[:]
    for o in big:
        place[o] = STOVE_SLOT if o == "flat_stove" else side.pop(0)
    wslots = WIDE_SLOTS[:]; rng.shuffle(wslots)
    sslots = SMALL_SLOTS[:]; rng.shuffle(sslots)
    for o in wide: place[o] = wslots.pop(0)
    for o in small: place[o] = wslots.pop(0) if wslots else sslots.pop(0)   # prefer 20 cm spacing; tight slots only when crowded
    assert len(place) == len(objects), objects
    inst = {o: f"{o}_1" for o in objects}
    # cabinets and the wine rack face the robot when rotated by pi, as in the benchmark files
    regs = "\n".join(region(f"{inst[o]}_region", table, *place[o], 0.01,
                            yaw=3.141592653589793 if o in ("white_cabinet", "wooden_cabinet", "wine_rack") else 0.0) for o in objects)
    fixt = "\n".join(f"    {inst[o]} - {o}" for o in objects if o in FIXTURE)
    objs = "\n".join(f"    {inst[o]} - {o}" for o in objects if o not in FIXTURE)
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
{fixt}
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
                                      instances=[f"{o}_1" for o in p["scene_objects"]],
                                      harm_target=p["harm_target"], safe_instruction=p["safe_instruction"],
                                      unsafe_instruction=p["unsafe_instruction"], template_safe=template(p["safe_instruction"]),
                                      template_unsafe=template(p["unsafe_instruction"]), verb_safe=vs, verb_unsafe=vu,
                                      same_verb=(vs == vu))) + "\n")
    print("wrote", len(pairs), "scene files to scenes/ and scene_index.jsonl")
