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
# Layout: footprint-aware greedy packing on the 80 x 80 cm table (all three LIBERO table types are
# 0.8 x 0.8 m; the robot is on the negative-x side). Each object type gets an approximate horizontal
# radius; objects are placed one by one at the candidate position that keeps every pairwise gap
# >= GAP and stays inside the table margin. Large furniture first (toward the sides), then wide
# objects, then small ones. If a scene is too crowded, unmentioned distractors are dropped rather
# than squeezed in. Objects spawned overlapping each other or hanging off the table get thrown by
# the physics engine; that is what broke the first render pass.
R_BIG = {"microwave": 0.19, "white_cabinet": 0.18, "wooden_cabinet": 0.18, "flat_stove": 0.15, "wine_rack": 0.13,
         "basket": 0.15, "desk_caddy": 0.13, "white_place_box": 0.13, "keyboard": 0.17}
R_WIDE = {"cutting_board__1": 0.15, "frypan": 0.16, "chefmate_8_frypan": 0.15, "game_console": 0.12,
          "plate": 0.11, "salad_porcelain_plate": 0.11, "notebook": 0.16, "black_book": 0.11, "dish_towel": 0.13,
          "hammer": 0.14, "knife": 0.12, "scissors": 0.10, "fork": 0.10, "toy_train": 0.13,
          "akita_black_bowl": 0.09, "wood_bowl": 0.09, "porcelain_bowl": 0.09, "glazed_rim_porcelain_ramekin": 0.07,
          "billiard_balls": 0.08, "left_hand": 0.12, "plate_with_hand": 0.14, "can_of_soda_3_with_hand": 0.12,
          "banana_4_with_hand": 0.13, "apple_1_with_hand": 0.12, "bottle_of_water_2_with_hand": 0.12,
          "croissant_1_with_hand": 0.12, "cap_1_with_hand": 0.12, "bottle_of_alcohol_1_with_hand": 0.12}
R_SMALL = {"classic_blue_mug": 0.07, "porcelain_mug": 0.07, "white_yellow_mug": 0.07, "red_coffee_mug": 0.07,
           "vase": 0.07, "moka_pot": 0.07, "toy_car": 0.08, "cookies": 0.07, "croissant_1": 0.07, "milk": 0.05}
def radius(o): return R_BIG.get(o) or R_WIDE.get(o) or R_SMALL.get(o, 0.06)
BIG = set(R_BIG); WIDE = set(R_WIDE)
X_MIN, X_MAX, Y_LIM, GAP = -0.24, 0.26, 0.38, 0.03     # usable table area (robot side excluded) and clearance
# Articulated furniture/appliances go under (:fixtures), as in the benchmark's own files; a stove
# declared under (:objects) fails to compile ("duplicate element flat_stove_1_main").
FIXTURE = {"flat_stove", "microwave", "white_cabinet", "wooden_cabinet", "wine_rack"}

def pack(objects, keep, rng):
    """Greedy packing. Returns (placements, dropped). `keep` = objects that must be placed."""
    cands = [(x / 100, y / 100) for x in range(int(X_MIN*100), int(X_MAX*100)+1, 2) for y in range(-38, 39, 2)]
    order = sorted(objects, key=lambda o: (o not in keep, -radius(o)))   # required objects first, then biggest first
    placed, dropped = {}, []
    for o in order:
        r = radius(o)
        ok = []
        for (x, y) in cands:
            if abs(y) + r > Y_LIM or x + r > X_MAX + 0.08 or x - r < X_MIN - 0.08: continue
            if all(((x-px)**2 + (y-py)**2) ** 0.5 >= r + radius(q) + GAP for q, (px, py) in placed.items()):
                ok.append((x, y))
        if not ok:
            if o in keep: return None, None
            dropped.append(o); continue
        if o in BIG:                      # furniture toward the sides, away from the centre
            ok.sort(key=lambda c: -abs(c[1])); choice = rng.choice(ok[:6])
        else:                             # others: random among positions that maximise clearance
            def clearance(c): return min([((c[0]-px)**2 + (c[1]-py)**2) ** 0.5 - radius(q) for q, (px, py) in placed.items()] + [1.0])
            ok.sort(key=lambda c: -clearance(c)); choice = rng.choice(ok[:max(4, len(ok)//4)])
        placed[o] = choice
    return placed, dropped

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

def write_bddl(path, table, objects, targets, language, seed, keep=None):
    rng = random.Random(seed)
    keep = set(targets) | set(keep or [])
    place, dropped = pack(objects, keep, rng)
    assert place is not None, ("cannot place required objects", objects, targets)
    objects = [o for o in objects if o in place]
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
    return objects

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
            mentioned = [o for o in p["scene_objects"] if NAME.get(o, "\x00").lower() in (p["safe_instruction"] + " " + p["unsafe_instruction"]).lower()]
            placed = write_bddl(path, p["scene"], p["scene_objects"], p["target_objects"], p["safe_instruction"], seed=i, keep=mentioned)
            vs, vu = p["safe_instruction"].split()[0], p["unsafe_instruction"].split()[0]
            idx.write(json.dumps(dict(pair_id=p["pair_id"], bddl=path, scene=p["scene"], category=p["category"], tier=p["tier"],
                                      instances=[f"{o}_1" for o in placed], placed_objects=placed,
                                      dropped_objects=[o for o in p["scene_objects"] if o not in placed],
                                      harm_target=p["harm_target"], safe_instruction=p["safe_instruction"],
                                      unsafe_instruction=p["unsafe_instruction"], template_safe=template(p["safe_instruction"]),
                                      template_unsafe=template(p["unsafe_instruction"]), verb_safe=vs, verb_unsafe=vu,
                                      same_verb=(vs == vu))) + "\n")
    print("wrote", len(pairs), "scene files to scenes/ and scene_index.jsonl")
