"""Build libero_distribution.json: which verbs / objects / scenes LIBERO contains and which of them pi0.5-LIBERO saw in fine-tuning.

Evidence for the training split: openpi's examples/libero/convert_libero_data_to_lerobot.py builds the fine-tuning dataset
(physical-intelligence/libero) from RAW_DATASET_NAMES = libero_{10,goal,object,spatial}_no_noops; libero_90 is not in it, and the
pi05_libero config trains on that dataset (src/openpi/training/config.py). The dataset's meta/tasks.jsonl (40 tasks, 1,693
episodes; saved in refs/) is the ground truth for the instruction TEXT the model saw. It uses rephrased wording ("pick up the
black bowl ...") that differs from the LIBERO repo's BDDL language ("Pick the akita black bowl ..."), so verbs and words come
from tasks.jsonl. Scenes/objects come from the BDDL files of the four suites (the dataset has no scene metadata; videos are
stored as images in the parquet files, which we did not download). Everything not in them is labelled OOD *relative to
LIBERO fine-tuning* only; pi0.5 base pre-training exposure is unknown.

Usage: python data_generation/extract_distribution.py   (needs cache/LIBERO, see README)
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from pull_libero_prompts import parse  # noqa: E402

LIBERO = HERE.parent / "cache/LIBERO/libero/libero"
TRAIN_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
VERBS = {  # lemma -> regex over lower-cased instruction
    "pick up": r"\bpick(?: up)?\b", "put": r"\bput\b", "place": r"\bplace\b", "open": r"\bopen\b", "close": r"\bclose\b",
    "push": r"\bpush\b", "turn on": r"\bturn on\b", "turn off": r"\bturn off\b", "stack": r"\bstack\b",
}


def snake(name):
    n = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower().replace("chefmate8", "chefmate_8")
    return {"rack": "wine_rack_alt"}.get(n, n)


def registry_objects():
    names = set()
    for f in (LIBERO / "envs/objects").glob("*.py"):
        for c in re.findall(r"^class (\w+)\(", f.read_text(), re.M):
            if not any(b in c for b in ("Base", "Articulated", "GoogleScanned", "Turbosquid", "TargetZone")) or c == "TurbosquidObjects" and False:
                names.add(snake(c))
    return names


def main():
    tasks = [parse(p) for p in sorted((LIBERO / "bddl_files").glob("*/*.bddl"))]
    for t in tasks:
        t["train"] = t["suite"] in TRAIN_SUITES
        t["layout"] = t["scene"]
    train = [t for t in tasks if t["train"]]
    hf = [json.loads(l)["task"].strip().lower() for l in open(HERE / "refs/pi_libero_tasks.jsonl")]
    assert len(hf) == 40, "expected the 40 fine-tuning tasks"

    verbs = {}
    for v, rx in VERBS.items():
        th = [x for x in hf if re.search(rx, x)]
        oh = [t for t in tasks if not t["train"] and re.search(rx, t["language"].lower())]
        verbs[v] = {"status": "ID" if th else "OOD", "train_tasks": len(th), "libero90_tasks": len(oh),
                    "example": (th or [t["language"] for t in oh] or [None])[0],
                    "evidence": "appears in the fine-tuning dataset's tasks.jsonl" if th else "absent from the 40 fine-tuning task strings (appears only in libero_90)"}

    reg = registry_objects()
    obj_train, obj_other = defaultdict(set), defaultdict(set)
    for t in tasks:
        for o in t["objects"] + t["fixtures"]:
            (obj_train if t["train"] else obj_other)[o].add(t["suite"])
    mention = lambda o, ts: any(o.replace("_", " ") in x or o.split("_")[-1] in x for x in hf)
    objects = {}
    for o in sorted(set(obj_train) | set(obj_other) | reg):
        in_train = o in obj_train
        objects[o] = {
            "status": "ID" if in_train else "OOD",
            "in_training_scenes": sorted(obj_train.get(o, [])), "in_libero90_scenes": o in obj_other,
            "in_libero_registry": o in reg or o in obj_train or o in obj_other,
            "named_in_training_instruction": mention(o, train),
            "evidence": ("present in the BDDL scene of >=1 fine-tuning task (visually seen, even if never named in an instruction)" if in_train
                         else "not present in any fine-tuning-suite scene; exists in LIBERO's asset registry / libero_90" if o in obj_other or o in reg else "unknown"),
        }

    scenes = {}
    for t in tasks:
        s = scenes.setdefault(t["layout"], {"status": "ID" if t["train"] else "OOD", "objects": set(), "fixtures": set(), "tasks": []})
        if t["train"]: s["status"] = "ID"
        s["objects"].update(t["objects"]); s["fixtures"].update(t["fixtures"]); s["tasks"].append(f'{t["suite"]}/{t["file"]}')
    for s in scenes.values():
        s["objects"], s["fixtures"] = sorted(s["objects"]), sorted(s["fixtures"])
        s["has_stove"] = "flat_stove" in s["fixtures"]
        s["evidence"] = "layout is the scene of a fine-tuning task" if s["status"] == "ID" else "layout only used by libero_90 tasks"

    out = {
        "training_distribution": {
            "used_for_labels": "instruction text: the 40 tasks in physical-intelligence/libero meta/tasks.jsonl (verified); scenes/objects: BDDL files of " + ", ".join(TRAIN_SUITES) + " (proxy for the dataset images)",
            "verified_from": ["openpi examples/libero/convert_libero_data_to_lerobot.py (RAW_DATASET_NAMES = libero_10/goal/object/spatial _no_noops)",
                              "openpi src/openpi/training/config.py: pi05_libero uses repo_id physical-intelligence/libero"],
            "not_verified": "dataset images were not downloaded, so scene/object exposure is inferred from the BDDL files of the matching suites",
            "caveat": "OOD means absent from LIBERO fine-tuning. pi0.5 base pre-training (and the PaliGemma backbone) may have seen these words/objects.",
            "n_training_tasks": len(train), "n_total_tasks": len(tasks), "libero_90_in_training": False,
        },
        "verbs": verbs, "objects": objects, "scenes": scenes,
        "training_instructions": sorted(hf), "training_dataset": json.load(open(HERE / "refs/pi_libero_info.json")) | {"features": "omitted"},
        "n_objects_registry": len(reg),
    }
    (HERE / "libero_distribution.json").write_text(json.dumps(out, indent=1))
    print("verbs", {k: v["status"] for k, v in verbs.items()})
    print("objects ID", sum(v["status"] == "ID" for v in objects.values()), "OOD", sum(v["status"] == "OOD" for v in objects.values()))
    print("scenes ID", [k for k, v in scenes.items() if v["status"] == "ID"])
    print("OOD scenes with stove", [k for k, v in scenes.items() if v["status"] == "OOD" and v["has_stove"]])


if __name__ == "__main__":
    main()
