"""Render scene images for the libero_safety pairs (CPU is fine; no model involved).

For every unique scene in a pairs file this script
  1. builds the BDDL: the benchmark BDDL as-is (config "benchmark_exact"), or a modified copy with the
     recorded `additions` / `arrangement` applied (config "composed"), saved under libero_safety/scenes/;
  2. renders it with LIBERO-Safety's OffScreenRenderEnv (MuJoCo) after a few no-op steps so objects settle;
  3. writes PNGs to cache/scene_images/ and a manifest (libero_safety/scene_manifest.json) with status per scene.

Setup (once):
  uv venv --python 3.10 .venv && source .venv/bin/activate
  uv pip install "numpy<2" mujoco==3.2.3 -e cache/LIBERO-Safety/third_party/robosuite-1.4 bddl==1.0.1 \
      easydict pyyaml imageio opencv-python-headless matplotlib gym==0.25.2 cloudpickle future h5py \
      termcolor numba scipy pillow pynput scikit-image remotezip
  cp -Rc cache/LIBERO/libero/libero/assets cache/libero_assets      # base object/scene assets (plain LIBERO)
LIBERO-Safety's own assets are one 10.7 GB zip on HuggingFace; `--fetch-assets` pulls only the folders a scene
actually needs out of that zip (HTTP range requests), into cache/libero_assets.

Usage:
  python libero_safety/render_scenes.py --pairs libero_safety/minimal_pairs_150.jsonl --examples 12   # quick look
  python libero_safety/render_scenes.py --pairs libero_safety/minimal_pairs_150.jsonl --all           # every scene
  python libero_safety/render_scenes.py --build-only ...      # only write the modified BDDLs (no simulator needed)
On a headless Linux box set MUJOCO_GL=egl (GPU) or osmesa (CPU) first.
"""
import argparse, hashlib, json, os, re, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LS = ROOT / "cache" / "LIBERO-Safety"
ASSETS = ROOT / "cache" / "libero_assets"
CFG = ROOT / "cache" / "libero_cfg"
SCENE_DIR = ROOT / "libero_safety" / "scenes"
IMG_DIR = ROOT / "cache" / "scene_images"
ZIP_URL = "https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/resolve/main/assets.zip"
CAMERAS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image", "front": "frontview_image"}  # front = human QA only


# ----------------------------------------------------------------------------- BDDL composition (pure text)
def block_span(text, key):
    """(start, end) of the balanced '(:key ...)' block."""
    i = text.index(f"(:{key}")
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "("
        depth -= text[j] == ")"
        if depth == 0:
            return i, j + 1
    raise ValueError(key)


def block_inner(text, key):
    s, e = block_span(text, key)
    return text[s + len(key) + 2 : e - 1]


def declared(text, key):
    """[(instance, type)] declared in the :objects / :fixtures block."""
    out, names = [], []
    for tok in re.findall(r"[^\s()]+", block_inner(text, key)):
        if tok == "-":
            names.append(None)
        elif names and names[-1] is None:
            names.pop()
            out += [(n, tok) for n in names]
            names = []
        else:
            names.append(tok)
    return out


def region_ranges(text, target):
    """Centres of regions whose :target is `target` (to place new objects away from them)."""
    cs = []
    for m in re.finditer(r"\((\w+)\s*\(:target\s+(\S+?)\)\s*\(:ranges\s*\(\s*\(([^)]*)\)", text):
        if m.group(2) == target:
            x0, y0, x1, y1 = map(float, m.group(3).split())
            cs.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return cs


def table_region(table, name, x, y, h=0.025):
    return (f"      ({name}\n          (:target {table})\n          (:ranges (\n              ({x-h:.3f} {y-h:.3f} {x+h:.3f} {y+h:.3f})\n            )\n          )\n"
            f"          (:yaw_rotation (\n              (0.0 0.0)\n          )\n          )\n      )\n")


def compose_bddl(base_text, scene):
    """Apply scene['additions'] and scene['arrangement'] to a BDDL; returns (text, notes)."""
    t = base_text
    objs, fixs = declared(t, "objects"), declared(t, "fixtures")
    inst = {}  # type -> first instance name (fixtures first, then objects)
    for n, ty in fixs + objs:
        inst.setdefault(ty, n)
    table = next(n for n, ty in fixs if ty.endswith("table"))
    new_decl, new_regions, init_add, init_drop, notes = [], [], [], set(), []
    taken = list(region_ranges(t, table))
    displaced, checks = [], []

    def new_instance(ty):
        k = 1 + max([int(n.rsplit("_", 1)[1]) for n, y in objs if y == ty and n.rsplit("_", 1)[1].isdigit()] + [0])
        name = f"{ty}_{k}"
        objs.append((name, ty)); inst.setdefault(ty, name); new_decl.append(f"    {name} - {ty}")
        return name

    def displace(region_name):
        """Objects the BDDL already puts in this region are moved to a free table spot (one region holds one object)."""
        for m in re.finditer(rf"\((?:On|In) (\S+) {re.escape(region_name)}\)", block_inner(t, "init")):
            init_drop.add(m.group(1)); displaced.append(m.group(1))

    def ensure_region(target, region):
        if not re.search(rf"\({region}\s*\(:target\s+{re.escape(target)}\)", t) and (target, region) not in [(a, b) for a, b, _ in new_regions]:
            new_regions.append((target, region, f"      ({region}\n          (:target {target})\n      )\n"))
        return f"{target}_{region}"

    def free_spot():
        cand = [(x / 100, y / 100) for x in range(-16, 9, 8) for y in range(-16, 17, 8)]  # inside the frame, away from the near table edge (objects drift)
        x, y = max(cand, key=lambda c: min([(c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 for p in taken] + [9]))
        taken.append((x, y))
        return x, y

    # additions that are not themselves arranged go on a free spot of the table first (they may be a target below)
    arranged = {a["object"] for a in scene["arrangement"]}
    for ad in scene["additions"]:
        if ad["object"] in arranged:
            continue
        name = new_instance(ad["object"])
        x, y = free_spot()
        reg = f"added_{name}_region"
        new_regions.append((table, reg, table_region(table, reg, x, y)))
        init_add.append(f"(On {name} {table}_{reg})")
        notes.append(f"{name} placed on table at ({x:.2f},{y:.2f})")

    placed = set()  # arranged additions are instantiated on first use
    for a in scene["arrangement"]:
        ty = a["object"]
        if ty in [x["object"] for x in scene["additions"]] and ty not in placed:
            name = new_instance(ty)
        else:
            name = inst[ty]; init_drop.add(name)  # relocate an object already in the scene
        placed.add(ty)
        if a.get("in") == "microwave":
            mw = inst["microwave"]
            reg = ensure_region(mw, "heating_region")
            displace(reg)
            init_add.append(f"(In {name} {reg})"); checks.append((name, reg))
            if not re.search(rf"\(Open {mw}\)", t):
                init_add.append(f"(Open {mw})")
        elif a.get("on") == "flat_stove":
            reg = ensure_region(inst["flat_stove"], "cook_region")
            displace(reg)
            init_add.append(f"(On {name} {reg})"); checks.append((name, reg))
        elif a.get("on"):
            init_add.append(f"(On {name} {inst[a['on']]})"); checks.append((name, inst[a["on"]]))
        else:
            raise ValueError(f"unsupported arrangement {a}")
    for name in displaced:
        x, y = free_spot()
        reg = f"moved_{name}_region"
        new_regions.append((table, reg, table_region(table, reg, x, y)))
        init_add.append(f"(On {name} {table}_{reg})")
        notes.append(f"{name} was in the target region; moved to table ({x:.2f},{y:.2f})")

    # splice (bottom-up so earlier offsets stay valid)
    s, e = block_span(t, "init")
    init = t[s:e]
    for n in init_drop:
        init = re.sub(rf"\s*\((?:On|In) {re.escape(n)} [^)]*\)", "", init)
    init = init[:-1].rstrip() + "\n" + "".join(f"    {x}\n" for x in init_add) + "  )"
    t = t[:s] + init + t[e:]
    s, e = block_span(t, "objects")
    t = t[: e - 1].rstrip() + "\n" + "\n".join(new_decl) + "\n  )" + t[e:]
    s, e = block_span(t, "regions")
    t = t[: e - 1].rstrip() + "\n" + "".join(r for _, _, r in new_regions) + "    )" + t[e:]
    return t, notes, checks


# ----------------------------------------------------------------------------- scene bookkeeping
def scene_key(sc):
    blob = json.dumps([sc["scene_id"], sc["additions"], sc["arrangement"]], sort_keys=True)
    return f"{sc['scene_id'].split('/')[-1][:48]}__{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def collect_scenes(pairs):
    out = {}
    for p in pairs:
        for side in ("harmless", "harmful"):
            sc = p[side]["scene"]
            out.setdefault(scene_key(sc), sc)
    return out


def scene_status(entry):
    """(ok, reasons) for a manifest entry: it rendered, every arranged object sits at its target, and nothing fell off the table, in every look (v0 + variants)."""
    if not entry or entry.get("status") != "ok":
        return False, ["render failed" if entry else "not rendered"]
    why = []
    for name, x in [("v0", entry)] + list(entry.get("variants", {}).items()):
        bad = [f"{c['object']}->{c['target']}" for c in x.get("placement_check", []) if c.get("ok") is False]
        if bad: why.append(f"{name}: placement failed {bad}")
        if x.get("fell_off"): why.append(f"{name}: fell off {x['fell_off']}")
    return not why, why


def build_scene_bddl(key, sc):
    src = LS / sc["bddl"]
    if sc["config"] == "benchmark_exact":
        return src, [], []
    text, notes, checks = compose_bddl(src.read_text(), sc)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    out = SCENE_DIR / f"{key}.bddl"
    out.write_text(text)
    return out, notes, checks


# ----------------------------------------------------------------------------- simulator side
def setup_env_vars():
    CFG.mkdir(parents=True, exist_ok=True)
    lib = LS / "libero" / "libero"
    (CFG / "config.yaml").write_text(
        f"benchmark_root: {lib}\nbddl_files: {lib/'bddl_files'}\ninit_states: {lib/'init_files'}\n"
        f"datasets: {LS/'libero'/'datasets'}\nassets: {ASSETS}\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(CFG)
    # LIBERO-Safety hard-codes <repo>/libero/libero/assets (ignores the config), so point that at our merged assets
    link = lib / "assets"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(os.path.relpath(ASSETS, lib))
    sys.path.insert(0, str(LS))
    # env_wrapper imports wand (ImageMagick) only for motion-blur corruptions; stub it so import works without it
    api, im = types.ModuleType("wand.api"), types.ModuleType("wand.image")
    api.library, im.Image = types.SimpleNamespace(MotionBlurImage=types.SimpleNamespace()), object
    api.library.MagickMotionBlurImage = types.SimpleNamespace()
    sys.modules.update({"wand": types.ModuleType("wand"), "wand.api": api, "wand.image": im})


_zip = None


def fetch_from_zip(missing_path):
    """Pull the folder containing `missing_path` out of the HF assets zip; True if something was fetched."""
    global _zip
    from remotezip import RemoteZip
    rel = Path(missing_path).relative_to(ASSETS.resolve())
    if _zip is None:
        _zip = RemoteZip(ZIP_URL)
        _zip.names = [i for i in _zip.infolist()]
    depth = 3 if rel.parts[0].startswith("new_objects") else 2  # new_objects/<name>/<variant>, others <suite>/<name>
    folder = "assets/" + "/".join(rel.parts[:depth]) + "/"
    members = [i for i in _zip.names if i.filename.startswith(folder)]
    if not members:
        return False
    new = 0
    for i in members:
        dest = ASSETS / i.filename[len("assets/"):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and not i.is_dir():
            dest.write_bytes(_zip.read(i.filename)); new += 1
    print(f"    fetched {new} new files ({len(members)} in folder): {folder}")
    return new > 0


def variant_params(v):
    """Visual variation for variant v>0 (v=0 is the untouched benchmark look). Deterministic in v only, so both sides of a pair, and every
    scene, get the same perturbation in the same variant: lighting never differs between harmless and harmful images."""
    if v == 0:
        return None
    import numpy as np
    r = np.random.RandomState(1000 + v)
    return {"diffuse_scale": float(r.uniform(0.55, 1.45)), "ambient_scale": float(r.uniform(0.6, 1.5)), "tint": [float(x) for x in r.uniform(0.88, 1.12, 3)],
            "light_shift_xy": [float(x) for x in r.uniform(-0.5, 0.5, 2)], "cam_shift": [float(x) for x in r.uniform(-0.03, 0.03, 3)], "fovy_delta": float(r.uniform(-3, 3))}


def apply_variant(env, base, p):
    """Set lighting + agentview camera on the MuJoCo model from the pristine `base` copy (so variants never stack)."""
    import numpy as np
    m = env.env.sim.model
    m.light_diffuse[:] = base["diffuse"]; m.light_ambient[:] = base["ambient"]; m.light_pos[:] = base["pos"]
    m.cam_pos[:] = base["cam_pos"]; m.cam_fovy[:] = base["fovy"]
    if p is None:
        return
    t = np.array(p["tint"])
    m.light_diffuse[:] = np.clip(base["diffuse"] * p["diffuse_scale"] * t, 0, 1.5)
    m.light_ambient[:] = np.clip(base["ambient"] * p["ambient_scale"] * t, 0, 1.5)
    m.light_pos[:, :2] = base["pos"][:, :2] + np.array(p["light_shift_xy"])
    cid = m.camera_name2id("agentview")
    m.cam_pos[cid] = base["cam_pos"][cid] + np.array(p["cam_shift"]); m.cam_fovy[cid] = base["fovy"][cid] + p["fovy_delta"]


def render_one(bddl, size, seed, settle, fetch, checks=(), variants=(0,), tries=1):
    """Render one scene once per variant (the simulator is built once). Returns ([(variant, images, placement_checks, fell_off, params)], seed_used).
    With tries>1 a scene whose placement check fails or where an object falls off is re-rendered with the next seed (different random object positions) until every
    variant is clean; the cleanest attempt is kept otherwise."""
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    for attempt in range(12):
        try:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size,
                                  camera_names=["agentview", "robot0_eye_in_hand", "frontview"])
            break
        except (FileNotFoundError, ValueError) as e:  # MuJoCo reports a missing mesh/texture as ValueError
            m = re.search(r"opening file '([^']+)'", str(e))
            f = e.filename if isinstance(e, FileNotFoundError) else (m.group(1) if m else None)
            missing = os.path.realpath(f) if f else ""
            if not (fetch and missing.startswith(str(ASSETS.resolve())) and fetch_from_zip(missing)):
                raise
    try:
        sm = env.env.sim.model
        base = {"diffuse": sm.light_diffuse.copy(), "ambient": sm.light_ambient.copy(), "pos": sm.light_pos.copy(), "cam_pos": sm.cam_pos.copy(), "fovy": sm.cam_fovy.copy()}
        best = None
        for sd0 in range(seed, seed + tries):
            out = []
            for v in variants:
                for sd in range(sd0, sd0 + 6):  # a placement can fail for an unlucky seed (RandomizationError)
                    try:
                        env.seed(sd)
                        obs = env.reset()
                        break
                    except Exception as e:
                        if "place all objects" not in str(e) or sd == sd0 + 5:
                            raise
                params = variant_params(v)
                apply_variant(env, base, params)
                for _ in range(settle):
                    obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                st = env.env.object_states_dict
                res = []
                for o, t in checks:  # did the arranged object really end up in/on its target (xy close, not below it)?
                    try:
                        po, pt = st[o].get_geom_state()["pos"], st[t].get_geom_state()["pos"]
                        res.append({"object": o, "target": t, "ok": bool(((po[:2] - pt[:2]) ** 2).sum() ** .5 < 0.1 and po[2] > pt[2] - 0.06),
                                    "xy_dist": round(float(((po[:2] - pt[:2]) ** 2).sum() ** .5), 3), "dz": round(float(po[2] - pt[2]), 3)})
                    except Exception as e:
                        res.append({"object": o, "target": t, "ok": None, "error": str(e)[:80]})
                zs = {n: float(x.get_geom_state()["pos"][2]) for n, x in st.items()}
                zmed = sorted(zs.values())[len(zs) // 2]  # tables differ in height, so judge against where most objects rest
                fell = [n for n, z in zs.items() if z < zmed - 0.25]  # fell to the floor
                out.append((v, {c: obs[k][::-1].copy() for c, k in CAMERAS.items()}, res, fell, params))  # flip vertically: MuJoCo images are upside-down
            bad = sum(bool(f) + sum(c.get('ok') is False for c in r) for _, _, r, f, _ in out)
            if best is None or bad < best[0]:
                best = (bad, out, sd0)
            if bad == 0:
                break
        return best[1], best[2]
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="libero_safety/minimal_pairs_150.jsonl")
    ap.add_argument("--all", action="store_true", help="render every unique scene")
    ap.add_argument("--examples", type=int, default=0, help="render the scenes of N pairs spread over hazard families")
    ap.add_argument("--build-only", action="store_true", help="only write modified BDDLs")
    ap.add_argument("--fetch-assets", action="store_true", help="download missing assets from the LIBERO-Safety zip")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=20, help="no-op steps before the snapshot")
    ap.add_argument("--manifest", default="libero_safety/scene_manifest.json")
    ap.add_argument("--variants", type=int, default=1, help="renders per scene: 1 = original look only; K>1 adds K-1 lighting/camera variants (v1..)")
    ap.add_argument("--tries", type=int, default=1, help="re-render a scene with the next seed (new random object positions) up to N times until placement checks pass and nothing falls off")
    ap.add_argument("--redo-bad", action="store_true", help="re-render only scenes that fail scene_status() (use with --tries)")
    ap.add_argument("--skip-done", action="store_true", help="skip scene/variant renders already 'ok' in the manifest")
    ap.add_argument("--montage-dir", default="results/scene_examples")
    a = ap.parse_args()

    pairs = [json.loads(l) for l in open(ROOT / a.pairs)]
    if a.examples and not a.all:  # round-robin over (changed component, hazard family), scene changes first
        by = {}
        for p in pairs:
            by.setdefault((p["changed_component"], p["hazard_family"]), []).append(p)
        order = sorted(by, key=lambda k: ({"scene": 0, "object": 1, "verb": 2}[k[0]], k[1]))
        chosen = []
        while len(chosen) < a.examples and any(by.values()):
            for k in order:
                if by[k] and len(chosen) < a.examples:
                    chosen.append(by[k].pop(0))
        pairs = chosen
    scenes = collect_scenes(pairs)
    print(f"{len(pairs)} pairs, {len(scenes)} unique scenes")

    man_path = ROOT / a.manifest
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    bddls = {}
    for k, sc in scenes.items():
        path, notes, checks = build_scene_bddl(k, sc)
        bddls[k] = path
        manifest[k] = {**manifest.get(k, {}), "scene_id": sc["scene_id"], "config": sc["config"], "bddl": str(path.relative_to(ROOT)),
                       "additions": [x["object"] for x in sc["additions"]], "arrangement": sc["arrangement"], "notes": notes, "checks": checks}
    if not a.build_only:
        setup_env_vars()
        import imageio
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        import time
        t0 = time.time()
        for n, (k, path) in enumerate(bddls.items(), 1):
            m = manifest[k]
            if a.redo_bad and (scene_status(m)[0] if m.get("status") else False):
                continue
            if a.redo_bad:
                m.pop("variants", None); m["status"] = "redo"
            done = {0} if (m.get("status") == "ok" and a.skip_done) else set()
            done |= {int(v[1:]) for v in m.get("variants", {}) if a.skip_done}
            todo = [v for v in range(a.variants) if v not in done]
            if not todo:
                continue
            try:
                outs, used_seed = render_one(path, a.size, a.seed, a.settle, a.fetch_assets, m.get("checks", []), todo, a.tries)
                for v, imgs, res, fell, params in outs:
                    files = {}
                    for cam, im in imgs.items():
                        f = IMG_DIR / (f"{k}__{cam}.png" if v == 0 else f"{k}__v{v}__{cam}.png")
                        imageio.imwrite(f, im)
                        files[cam] = str(f.relative_to(ROOT))
                    if v == 0:
                        m.update(status="ok", images=files, size=a.size, seed=used_seed, placement_check=res, fell_off=fell)
                    else:
                        m.setdefault("variants", {})[f"v{v}"] = {"images": files, "params": params, "placement_check": res, "fell_off": fell}
                    if v == 0 and (any(not r["ok"] for r in res) or fell):
                        print(f"   {k}: PLACEMENT? {[r for r in res if not r['ok']]} FELL OFF: {fell}")
                print(f"[{n}/{len(bddls)}] ok   {k}   ({time.time() - t0:.0f}s elapsed)", flush=True)
            except Exception as e:  # keep going; the manifest records why a scene failed
                m.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
                print(f"[{n}/{len(bddls)}] FAIL {k}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if n % 10 == 0:
                man_path.write_text(json.dumps(manifest, indent=1))
    man_path.write_text(json.dumps(manifest, indent=1))
    ok = sum(m.get("status") == "ok" for m in manifest.values())
    print(f"manifest: {ok}/{len(manifest)} scenes rendered -> {a.manifest}")
    if a.examples and not a.build_only:
        make_montages(pairs, manifest, ROOT / a.montage_dir)


def make_montages(pairs, manifest, outdir):
    """Side-by-side harmless | harmful image per pair, captioned with the instruction and safety facts."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    outdir.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        fig, ax = plt.subplots(3, 2, figsize=(6.4, 9.2))
        for col, side in enumerate(("harmless", "harmful")):
            m = manifest.get(scene_key(p[side]["scene"]), {})
            for row, cam in enumerate(("agentview", "wrist", "front")):
                x = ax[row][col]; x.set_xticks([]); x.set_yticks([])
                if col == 0:
                    x.set_ylabel(cam, fontsize=8)
                if m.get("status") == "ok":
                    x.imshow(Image.open(ROOT / m["images"][cam]))
                else:
                    x.text(0.5, 0.5, "not rendered", ha="center")
            ax[0][col].set_title(f"{side.upper()}\n" + "\n".join(textwrap.wrap(p[side]["instruction"], 38)), fontsize=9)
        fig.suptitle(f"{p['pair_id']} · changed: {p['changed_component']} · {p['harm_type']} · {p['semantic_level']} · {p['hazard_family']}\n"
                     + "\n".join(textwrap.wrap(p["harm_mechanism"], 100)), fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(outdir / f"{p['pair_id']}.png", dpi=90)
        plt.close(fig)
    print(f"montages -> {outdir}")


if __name__ == "__main__":
    main()
