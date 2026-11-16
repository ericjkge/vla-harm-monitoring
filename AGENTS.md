# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Restarted project: test whether PaliGemma, `pi05_base` and `pi05_libero` internally represent physical harm (linear probes on activations, 500 LIBERO-only minimal instruction pairs, split 300/100/100), as the basis for an inference-time monitor. Constraints: **LIBERO only (no LIBERO-Safety), no fine-tuning**. Full plan: see README.md and the saved project-plan memory. Extension (openvla-oft, non-linear probes, interventions) goes in `extension/`.

## Layout

`src/` library code, `scripts/` entry points, `configs/`, `extension/`, `results/` (small committed outputs), `cache/` and `logs/` (gitignored). `Old/` is the archived previous project (execution grid, LIBERO-Safety, Kempner SLURM scripts); do not build on it or move it back without asking. Git still shows its files as deleted at their old root paths.

## Pair generation

Two independent pipelines, both: one changed component per pair (verb / object / scene), safety facts in a hand-written `lexicon.py`, validator re-derives labels from the lexicon (no LLM judge).

- `libero/` (first pilot, plain-LIBERO ID/OOD relative to the pi05_libero fine-tuning split): `python libero/extract_distribution.py && python libero/generate_pairs.py --per-condition 20 && python libero/validate_pairs.py --report libero/dataset_report.md`. Condition 1 is intentionally empty; do not pad it.
- `libero_safety/` (current): ID means "appears anywhere in LIBERO-Safety" (clone to `cache/LIBERO-Safety` first: `git clone --depth 1 https://github.com/LIBERO-SAFETY/LIBERO-Safety.git cache/LIBERO-Safety`). Run `python libero_safety/extract_components.py && python libero_safety/generate_pairs.py && python libero_safety/validate_pairs.py --report libero_safety/dataset_report.md`. `--relax-caps` writes `*_relaxed.jsonl` (reaches 150, more same-mechanism variants). Scenes are benchmark BDDL scenes, sometimes "composed" (objects added from other benchmark scenes, recorded with sources); images come from `libero_safety/render_scenes.py` (CPU MuJoCo, runs in `.venv` Python 3.10: setup commands in its docstring; `--fetch-assets` pulls only needed folders out of the 10.7 GB LIBERO-Safety assets zip into `cache/libero_assets`; writes `libero_safety/scenes/*.bddl`, `scene_manifest.json` with placement checks, PNGs in `cache/scene_images/`; `--examples N` makes montages in `results/scene_examples/`). Check manifest `placement_check`/`fell_off` before using a scene. `python libero_safety/generate_pairs.py --target 500` builds `minimal_pairs_500.jsonl`: the 129 original pairs are kept and labelled `batch: original` (the rest `expansion`) so you can compare the two for memorization; each pair has `split` (train/val/test, ~60/20/20) and `group` (hazard family + hazardous object; a group never spans splits). `--explicit-frac` is a request, capped by the rendered-clean explicit candidates (only smash/stab/push-off/pour verbs exist in the benchmark); every pair carries `scene_ok` from the render manifest and pairs on failed scenes are replaced from the pool; `render_scenes.py --tries N --redo-bad` re-renders bad scenes with new seeds. `render_scenes.py --variants 3` renders the original look plus 2 lighting/camera variants per scene (identical perturbation on both sides of a pair; images `*__v1__*.png`, manifest `variants`); `--skip-done` resumes. Scene-only pairs cannot be explicit_harm (text is identical). Never add verbs/objects outside `libero_safety_components.json`.

## Commands

No build/lint/test setup yet. Pull the LIBERO prompts:
`git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git cache/LIBERO && python scripts/pull_libero_prompts.py`

## Data notes

LIBERO prompts come from the `:language` field of the BDDL files in `cache/LIBERO/libero/libero/bddl_files/{libero_10,libero_90,libero_goal,libero_object,libero_spatial}`. Only libero_90/libero_10 file names carry a `KITCHEN/LIVING_ROOM/STUDY_SCENE<n>` prefix; goal/object/spatial each use a single fixed scene layout.
