# Do VLMs and VLAs represent physical harm?

Refusal training targets explicit harmful text, not physical harm implied by an action in a scene, and most VLAs output only actions, so they cannot refuse. We test whether harm is linearly readable from internal activations at three stages (PaliGemma VLM -> `pi05_base` -> `pi05_libero`), so that an activation monitor could flag unsafe prompts at inference time. Extension: openvla-oft.

- **Data:** 500 minimal (harmless, harmful) instruction pairs on LIBERO scenes/objects only (LIBERO-Safety is not used). Split 300/100/100.
- **Method:** linear probe per model stage; if separable, test-time intervention; if not, non-linear probes.
- **No fine-tuning** for now.

## Layout
| Path | Purpose |
|---|---|
| `src/` | library code (data, activation extraction, probes) |
| `scripts/` | entry points, e.g. `pull_libero_prompts.py` |
| `configs/` | experiment configs |
| `libero_safety/` | second dataset: 150-pair recombination of LIBERO-Safety components (`extract_components.py` -> `libero_safety_components.json`, `generate_pairs.py`, `validate_pairs.py`, `dataset_report.md`, `render_scenes.py` -> composed BDDLs + scene images); ID = appears anywhere in the LIBERO-Safety benchmark |
| `libero/` | minimal-pair dataset: `extract_distribution.py` -> `libero_distribution.json`, `generate_pairs.py` -> `minimal_pairs_pilot.jsonl`, `validate_pairs.py` (+ `dataset_report.md`); `lexicon.py` holds safety facts |
| `extension/` | openvla-oft, non-linear probes, interventions |
| `results/` | small outputs (e.g. `libero_prompts.jsonl`) |
| `cache/`, `logs/` | gitignored (LIBERO clone, model downloads, logs) |
| `Old/` | previous project (execution grid, LIBERO-Safety); archived, not part of the new plan |

## Getting LIBERO prompts
```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git cache/LIBERO
python scripts/pull_libero_prompts.py      # -> results/libero_prompts.jsonl (130 tasks)
```
