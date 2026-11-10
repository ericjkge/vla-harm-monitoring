"""Shared helpers for the cluster scripts: environment creation, model loading, prompt format."""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silences a fork warning from the video writer
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSP = os.path.join(REPO, "semantic_safety_pairs")
ROOT = os.environ.get("VLA_HARM_ROOT", os.path.expanduser("~/vla-harm"))
BENCH_BDDL = os.path.join(ROOT, "LIBERO-Safety/libero/libero/bddl_files")
# LIBERO's editable install does not reliably expose the `libero` package outside its own folder,
# so put the clone on the path explicitly.
import sys
sys.path.insert(0, os.path.join(ROOT, "LIBERO-Safety"))

def make_env(bddl_path, seed=0, size=256):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=size, camera_widths=size)
    env.seed(seed); return env

def settle(env, steps=10):
    obs = env.reset()
    for _ in range(steps): obs, *_ = env.step([0.]*6 + [-1.])
    return obs

def frame(obs):
    from PIL import Image
    return Image.fromarray(obs["agentview_image"][::-1].copy())

def positions(env, insts):
    sim = env.env.sim; out = {}
    for inst in insts:
        for cand in (f"{inst}_main", inst):
            if cand in sim.model.body_names:
                out[inst] = np.array(sim.data.body_xpos[sim.model.body_name2id(cand)]); break
    return out

def load_vla(model_id="openvla/openvla-7b-finetuned-libero-10", four_bit=False):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if four_bit:
        from transformers import BitsAndBytesConfig
        import transformers.modeling_utils as mu
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        _to = mu.PreTrainedModel.to; mu.PreTrainedModel.to = lambda self, *a, **k: self
        try: vla = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16, **kw)
        finally: mu.PreTrainedModel.to = _to
        for _, b in vla.named_buffers():
            if b.device.type != "cuda": b.data = b.data.to("cuda")
        dtype = torch.float16
    else:
        vla = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw).to("cuda"); dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    vla.eval(); return vla, proc, dtype

def prompt_for(instr): return f"In: What action should the robot take to {instr.lower()}?\nOut:"

def predict_action(vla, proc, dtype, img, instr, unnorm_key="libero_10"):
    import torch
    with torch.no_grad():
        inputs = proc(prompt_for(instr), img).to("cuda", dtype=dtype)
        return vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

def hidden_states(vla, proc, dtype, img, instr):
    """[n_layers+1, d] last-token hidden state at every layer (prefill only)."""
    import torch
    with torch.no_grad():
        inputs = proc(prompt_for(instr), img).to("cuda", dtype=dtype)
        out = vla(**inputs, output_hidden_states=True)
        return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()

def load_jsonl(p): return [json.loads(l) for l in open(p)]
