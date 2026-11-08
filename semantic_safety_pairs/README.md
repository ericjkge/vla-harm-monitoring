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
| `ssr_pilot_colab.ipynb` | Colab notebook: render the 15 SSR scenes, run OpenVLA on safe+unsafe prompts, check linear separability |

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

## Suggested next steps

1. Run `ssr_pilot_colab.ipynb` on an A100 Colab. If layer-wise LOO accuracy is near chance on
   the 30 benchmark prompts, the 500 pairs are needed; if high, that is the signal.
2. Render the 500 pairs' scenes, extract hidden states, train linear probes with a
   template-held-out split, evaluate on the 15 SSR tasks + the other external benchmarks.
