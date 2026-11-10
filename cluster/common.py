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

# Stock robosuite 1.4.0 arm controller, which original LIBERO (and so OpenVLA's training data) uses. LIBERO-Safety's
# bundled robosuite changes exactly these fields (output_max 2 m and 2 rad per unit action instead of 0.05 m and
# 0.5 rad, kp 750 instead of 150), i.e. the arm moves 40x farther per command. It is the only code difference
# between their robosuite and upstream. A policy trained on original LIBERO flails under their values.
STOCK_OSC_POSE = dict(output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5], output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                      kp=150, kp_limits=[0, 300])

def make_env(bddl_path, seed=0, size=256, stock_controller=True):
    """stock_controller=True for policies trained on original LIBERO (OpenVLA); False keeps LIBERO-Safety's own
    controller, for policies trained on LIBERO-Safety data."""
    import robosuite
    from libero.libero.envs import OffScreenRenderEnv
    orig = robosuite.load_controller_config
    def patched(*a, **k):
        cfg = orig(*a, **k)
        if stock_controller and cfg.get("type") == "OSC_POSE": cfg.update(STOCK_OSC_POSE)
        return cfg
    robosuite.load_controller_config = patched  # the env wrapper looks this up as robosuite.load_controller_config
    try: env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=size, camera_widths=size)
    finally: robosuite.load_controller_config = orig
    if stock_controller:
        c = env.env.robots[0].controller
        assert c is None or abs(float(np.max(c.output_max)) - 0.5) < 1e-6 and abs(float(c.output_max[0]) - 0.05) < 1e-6, f"controller not patched: {c.output_max}"
    env.seed(seed); return env

def settle(env, steps=10):
    obs = env.reset()
    for _ in range(steps): obs, *_ = env.step([0.]*6 + [-1.])
    return obs

def frame(obs, crop_scale=0.9):
    """Model input, matching OpenVLA's own LIBERO evaluation: rotate the raw render 180 degrees (both axes, as in
    the training data), resize to 224 with Lanczos, then take the central 90%-area crop and resize back (the
    LIBERO fine-tunes were trained with random-crop augmentation, so they expect a center crop at test time)."""
    from PIL import Image
    import io
    buf = io.BytesIO(); Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).save(buf, format="JPEG", quality=95)
    img = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)  # JPEG round trip, as in the training data pipeline
    if crop_scale:
        m = 224 * (1 - crop_scale ** 0.5) / 2
        img = img.crop((m, m, 224 - m, 224 - m)).resize((224, 224), Image.BILINEAR)
    return img

def view(obs):
    """Upright frame for videos and for humans to look at (not what the model sees)."""
    return obs["agentview_image"][::-1].copy()

def env_action(act):
    """OpenVLA outputs gripper in [0, 1] with 1 = open; the simulator wants -1 = open, +1 = close.
    Same post-processing as OpenVLA's LIBERO evaluation: rescale to [-1, 1], binarize, flip the sign."""
    a = np.array(act, dtype=np.float64).copy()
    a[6] = 1.0 if a[6] < 0.5 else -1.0
    return a

def quat2axisangle(quat):
    """robosuite (x, y, z, w) quaternion -> axis-angle, as in openpi's LIBERO example."""
    q = np.array(quat, dtype=np.float64); q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if np.isclose(den, 0.0) else q[:3] * 2.0 * np.arccos(q[3]) / den

class Pi05Client:
    """pi0.5 served by openpi's scripts/serve_policy.py. Inputs follow openpi's examples/libero/main.py exactly:
    third-person and wrist images rotated 180 degrees and resized to 224, 8-d state (end-effector position,
    axis-angle orientation, gripper joints), raw prompt. The model returns a chunk of actions; the first `replan`
    are executed before asking again. Actions go to the simulator unchanged (no gripper post-processing)."""
    def __init__(self, host="127.0.0.1", port=8000, replan=5):
        import collections
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)  # waits until the server is up
        self.replan, self.plan = replan, collections.deque()
    def reset(self): self.plan.clear()
    @staticmethod
    def _img(obs, key):
        from PIL import Image
        return np.asarray(Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    def act(self, obs, instr):
        if not self.plan:
            chunk = self.client.infer({
                "observation/image": self._img(obs, "agentview_image"),
                "observation/wrist_image": self._img(obs, "robot0_eye_in_hand_image"),
                "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": str(instr)})["actions"]
            assert len(chunk) >= self.replan, f"policy returned {len(chunk)} actions, need {self.replan}"
            self.plan.extend(chunk[:self.replan])
        return np.array(self.plan.popleft(), dtype=np.float64)

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
    """[n_layers+1, d] hidden state at every layer, read at the position the first action token is predicted from.
    predict_action appends the empty token 29871 after "Out:" before generating (as in training), so the same token
    is appended here; without it the read position would be one token earlier than where the model decides."""
    import torch
    with torch.no_grad():
        inputs = proc(prompt_for(instr), img).to("cuda", dtype=dtype)
        ids, am = inputs["input_ids"], inputs["attention_mask"]
        if ids[0, -1].item() != 29871:
            inputs["input_ids"] = torch.cat([ids, torch.full((1, 1), 29871, dtype=ids.dtype, device=ids.device)], dim=1)
            inputs["attention_mask"] = torch.cat([am, torch.ones((1, 1), dtype=am.dtype, device=am.device)], dim=1)
        out = vla(**inputs, output_hidden_states=True)
        return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()

def load_jsonl(p): return [json.loads(l) for l in open(p)]
