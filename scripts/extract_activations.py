"""Extract per-layer residual-stream activations for every sample in probe_table.jsonl.

One forward pass through the VLM backbone (prefix only: image + instruction, no action head),
batch size 1. Saves, for each of the L+1 hidden states (embeddings + every block):
  last_instr : hidden state at the last instruction token
  mean_instr : mean over instruction tokens
  mean_img   : mean over image tokens
  last_tok   : hidden state at the final position (the "\n" the processor appends)

Usage (GPU box, after `huggingface-cli login` and accepting the PaliGemma license):
  python scripts/extract_activations.py --model paligemma --out cache/acts/paligemma.npz
  python scripts/extract_activations.py --model paligemma --variant v1 --out cache/acts/paligemma_v1.npz
  python scripts/extract_activations.py --model paligemma --no-text --out cache/acts/paligemma_imgonly.npz   # image-only control
  python scripts/extract_activations.py --model paligemma --limit 20 --out cache/acts/smoke.npz              # smoke test

Rows keep the order of probe_table.jsonl; the npz stores pair_id / label / split / group so
scripts/train_probes.py can align everything. pi05_base / pi05_libero: see load_pi05() (not implemented yet).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_rows(table, variant, view, limit):
    rows = [json.loads(l) for l in open(table)]
    if limit: rows = rows[:limit]
    for r in rows:
        imgs = r["images"] if variant == "orig" else r["variants"][variant]
        r["image_path"] = str(ROOT / imgs[view])
    return rows


# ----------------------------------------------------------------------------- PaliGemma
def load_paligemma(hf_id, device):
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(hf_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(device).eval()
    n_img = model.config.image_seq_length if hasattr(model.config, "image_seq_length") else 256
    img_id = model.config.image_token_index

    @torch.no_grad()
    def run(image, text):
        """-> dict of [L+1, d] float32 numpy arrays."""
        inp = proc(text=text, images=image, return_tensors="pt").to(device)
        ids = inp["input_ids"][0]
        out = model(**inp, output_hidden_states=True, return_dict=True)
        hs = torch.stack([h[0] for h in out.hidden_states]).float()  # [L+1, T, d]
        is_img = ids == img_id
        img_pos = torch.nonzero(is_img).flatten()
        if len(img_pos) != n_img: raise RuntimeError(f"expected {n_img} image tokens, got {len(img_pos)}")
        feats = {"mean_img": hs[:, img_pos].mean(1), "last_tok": hs[:, -1]}
        if text:
            # processor layout: [image]*n_img, <bos>, instruction tokens, "\n"
            k = len(proc.tokenizer(text, add_special_tokens=False)["input_ids"])
            start = int(img_pos[-1]) + 2  # skip <bos>
            pos = torch.arange(start, start + k, device=device)
            tok = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids[pos].tolist() != tok: raise RuntimeError(f"instruction token mismatch for {text!r}")
            feats["last_instr"] = hs[:, pos[-1]]
            feats["mean_instr"] = hs[:, pos].mean(1)
        return {k: v.cpu().numpy() for k, v in feats.items()}

    return run


# ----------------------------------------------------------------------------- pi0.5
def load_pi05(name, device):
    """TODO (needs the GPU box + openpi): build a run(image, text) with the same return contract as
    load_paligemma. Notes for whoever fills this in:
      * checkpoints: gs://openpi-assets/checkpoints/{pi05_base,pi05_libero}
      * only run the PaliGemma prefix (images + tokenized prompt), not the action expert / flow steps
      * hook the residual stream after every Gemma block of the VLM backbone; return [L+1, d] per feature
      * use identical inputs for all three models (same image, same prompt string)
      * openpi's LIBERO policy expects images in its own orientation/size (224x224); check whether the
        rendered scene PNGs need the usual 180-degree flip before feeding them in
    """
    raise NotImplementedError("pi05 extraction not implemented yet; see docstring of load_pi05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["paligemma", "pi05_base", "pi05_libero"], required=True)
    ap.add_argument("--hf-id", default="google/paligemma-3b-pt-224", help="PaliGemma checkpoint")
    ap.add_argument("--table", default=str(ROOT / "results/dataset_review/probe_table.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="agentview", choices=["agentview", "wrist", "front"])
    ap.add_argument("--variant", default="orig", help="orig, v1 or v2 (lighting/camera variants)")
    ap.add_argument("--no-text", action="store_true", help="image-only control: empty prompt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from PIL import Image
    rows = load_rows(a.table, a.variant, a.view, a.limit)
    run = load_paligemma(a.hf_id, a.device) if a.model == "paligemma" else load_pi05(a.model, a.device)

    acc, t0 = {}, time.time()
    for i, r in enumerate(rows):
        f = run(Image.open(r["image_path"]).convert("RGB"), "" if a.no_text else r["instruction"])
        for k, v in f.items():
            acc.setdefault(k, []).append(v.astype(np.float16))
        if i % 50 == 0: print(f"{i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, **{k: np.stack(v) for k, v in acc.items()},  # each [N, L+1, d]
        pair_id=np.array([r["pair_id"] for r in rows]), label=np.array([r["label"] for r in rows]),
        split=np.array([r["split"] for r in rows]), group=np.array([r["group"] for r in rows]),
        meta=json.dumps(dict(model=a.model, hf_id=a.hf_id, view=a.view, variant=a.variant, no_text=a.no_text)))
    print("saved", a.out, {k: np.stack(v).shape for k, v in acc.items()})


if __name__ == "__main__":
    sys.exit(main())
