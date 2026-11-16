# Semantic-safety instruction pairs for VLA probing

Matched (safe, unsafe) language-instruction pairs for probing whether a frozen
Vision-Language-Action model internally separates harmful from harmless intent.
Each pair shares a scene and target object(s); only the instruction's intent differs.

## Files

| File | What it is |
|---|---|
| `generate_pairs.py` | Template-based generator. `python3 generate_pairs.py --n 500 --seed 0 --out pairs` |
| `pairs.jsonl` / `pairs.csv` | 500 generated pairs (seed 0) |
| `pairs_libero_ssr_heldout.txt` | The 15 LIBERO-Safety Semantic Safety Reasoning (SSR) instructions, held out from training |
| `libero_ssr_matched_pairs.jsonl` | A hand-written safe counterpart for each of the 15 SSR tasks, on the same scene |
| `ssr_pilot_colab.ipynb` | Colab notebook (T4-tested): render the 15 SSR scenes, run OpenVLA on safe+unsafe prompts, probe; phase-2 cells 10-11 run the generated pairs |
| `ssr_scene_objects.json` | Object list of each of the 15 SSR scenes (from their BDDL files) |
| `match_pairs_to_scenes.py` | Assigns generated pairs to SSR scenes whose objects they mention; derives template keys and a same-verb flag |
| `ssr_scene_pairs.jsonl` | 96 generated pairs matched to the 15 scenes (24 strictly grounded, 72 targets-only) |
| `make_scene_bddls.py` | Writes one LIBERO BDDL scene per pair with exactly the objects it names |
| `scenes/` | 500 scene files (`<pair_id>.bddl`), ready to render with `OffScreenRenderEnv` |
| `scene_index.jsonl` | pair -> scene file, plus template keys and same-verb flag |
| `sim_object_registry.json` | The 1,420 object type names the simulator accepts (all 61 we use are in it) |
| `results/` | Pilot outputs: `ssr_pilot_records.pkl` (30 x 33 layers x 4096 last-token activations), `ssr_pilot_probes.png` |

## Pair schema (`pairs.jsonl`)

| Field | Meaning |
|---|---|
| `pair_id` | `ssp_0000` ... |
| `category` | human_injury, human_discomfort, property_destruction, liquid_damage, fire_heat, appliance_misuse, ingestion, fragile_stacking, hazard_placement, property_misuse |
| `tier` | L0 explicit harm / L1 physical-common-sense violation / L2 contextual trap (mirrors LIBERO-Safety SSR) |
| `harm_target` | human / property / environment |
| `scene` | kitchen_table / living_room_table / study_table (LIBERO table types) |
| `scene_objects` | LIBERO-Safety simulator object types in the scene (shared by both instructions) |
| `target_objects` | Objects the instruction acts on |
| `safe_instruction`, `unsafe_instruction` | The matched pair |
| `rationale` | Why the unsafe one is unsafe |

Object names are LIBERO-Safety sim types (`knife`, `plate_with_hand`, `game_console`, ...) so each
pair can later be turned into a BDDL scene with the repo's `scripts/create_bddl.py`.

## Why not use LIBERO-Safety's own data?

Its 19.7k demonstrations are collision-free trajectories for the four *physical* suites only.
There are no unsafe trajectories, no safe/unsafe pairs, and no demonstrations at all for the
semantic track. The semantic track (`reasoning_safety`, 15 BDDL tasks, 5 per tier) is
evaluation-only and is scored by refusal rate on reasoning models, not VLAs.

## Leakage guard

Any generated unsafe instruction that is a near-duplicate of one of the 15 SSR benchmark
instructions is dropped, so the SSR track stays a clean held-out test. Template *structure*
is allowed to overlap (e.g. other objects in the microwave); that is the generalization we want
to test.

## Known limitations (to fix before training a real probe)

- Template-generated: safe instructions repeat more than unsafe ones (≈300 vs ≈440 unique).
  A probe could key on surface form. Add paraphrases (LLM or by hand) and check that the
  probe survives a held-out-template split.
- No visual grounding yet: scenes are object lists, not rendered frames. Instantiate as BDDL
  files and render with `OffScreenRenderEnv` before extracting activations.
- Category balance is uneven (human_injury largest). Adjust `--max_per_category` or add templates.
- Several L2 items are only unsafe by context (e.g. "the banana fell on the floor");
  decide whether the VLA is even expected to parse that.

## Pilot result (2026-11-08, OpenVLA-7B LIBERO-10 finetune, 4-bit on a T4)

On the 15 SSR tasks with a hand-written safe twin each, a difference-of-means probe evaluated
leave-one-pair-out orders the held-out unsafe prompt above its safe twin in 13-15 of 15 pairs at
every layer from 2 to 30 (15/15 at several layers). It still works on the pairs where the unsafe
prompt is the longer one, so it is not reading prompt length. Plain logistic regression found
nothing at n=30 in 4096-d, so probe method matters at this scale. Closed-loop rollouts were not run:
free Colab cannot hold the simulator and the model at once.

## Phase 2 (ready to run): verb-vs-harm test

`ssr_scene_pairs.jsonl` puts 96 generated pairs onto the already-rendered scenes. Notebook cells
10-11 extract their activations and run three checks: leave-one-template-out, the same-verb subset
(e.g. "put X next to Y" vs "put X on top of Y"), and train-on-generated / test-on-benchmark with
the test scene's pairs held out. If the same-verb subset separates, the direction is about harm,
not verb identity. Only 24 pairs are strictly grounded in the image; the rest mention one absent
object. Rendering more scenes from `pairs.jsonl` is the way to grow this.

## Phase 3 (ready to run): all 500 pairs on their own scenes

Every pair now has a scene file with exactly the objects its instructions name, so the
absent-object confound of phase 2 is gone. Notebook cells 12-14: render all 500 scenes with no
model loaded (~1 h on free Colab, resumable), then run the model on the saved frames (~25 min),
then evaluate leave-one-template-out, the same-verb subset, per-category and per-tier accuracy,
and train-on-500 / test-on-15-benchmark. Scene layout is footprint-aware greedy packing on the
80 x 80 cm table: each object type has an approximate radius, nothing overlaps, nothing hangs off
the edge, furniture/appliances are `:fixtures` as in the benchmark's files. When a scene is too
crowded, unmentioned distractors are dropped; `scene_index.jsonl` records `placed_objects` and
`dropped_objects` per pair (68 of 500 dropped one or two). Cell 12 records `_warnings.json` for any scene where an
object moved more than 8 cm while settling, which is the signature of objects spawned overlapping
and flung off the table. Run `12-test` first: it renders 12 scenes in ~5 min and shows them.

Lessons from the first render passes (2026-11-09): a stove under `:objects` fails to compile; a
10 cm grid with 25-45 cm objects interpenetrates and MuJoCo throws them off the table (~1/3 of
frames had missing objects); fixed side slots at 34 cm put 30-40 cm furniture over the edge of the
80 cm table and into the first grid column (10 of 12 test scenes flagged). Free Colab: render on a CPU runtime (no GPU quota used), frames are
zipped and downloaded every 50 scenes because Drive mounting is blocked on the Harvard account.

## Cluster

All of the above, plus closed-loop rollouts, as SLURM scripts: see `../cluster/README.md`.

## Suggested next steps

1. Run `ssr_pilot_colab.ipynb` on an A100 Colab. If layer-wise LOO accuracy is near chance on
   the 30 benchmark prompts, the 500 pairs are needed; if high, that is the signal.
2. Render the 500 pairs' scenes, extract hidden states, train linear probes with a
   template-held-out split, evaluate on the 15 SSR tasks + the other external benchmarks.
