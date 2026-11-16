"""Extract the component inventory (verbs, objects, scenes, suites, levels) of LIBERO-Safety -> libero_safety_components.json.

    git clone --depth 1 https://github.com/LIBERO-SAFETY/LIBERO-Safety.git cache/LIBERO-Safety
    python libero_safety/extract_components.py

Scope = the 205 registry tasks in libero/libero/benchmark/vla_safety_task_map.py (5 safety suites x 5 tasks x 3 levels = 75, plus
libero_10/90/spatial/object/goal = 130). The repo's bddl_files also hold ~8,000 augmented variants (libero_mix, "_add_N", "copy",
LLM paraphrases of the instruction). They are NOT part of the registry; their extra vocabulary is listed separately as
`variant_only` and is never used for generation.
"""
import ast, json, re, subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent / "cache/LIBERO-Safety"
LIB = REPO / "libero/libero"
SAFETY_SUITES = ["affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human", "reasoning_safety"]
LIBERO_SUITES = ["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"]
VARIANT = re.compile(r"(_add_\d+|copy)")

# surface verb -> (normalized group, normalization confidence). Surface forms are kept; the group is only for reporting.
VERB_GROUPS = {
    "pick up": ("pick up", "high"), "pick": ("pick up", "high"), "grab": ("pick up", "medium"), "grasp": ("pick up", "medium"),
    "get": ("pick up", "low"), "retrieve": ("pick up", "medium"), "put": ("place", "high"), "place": ("place", "high"),
    "move": ("place", "medium"), "bring": ("hand over", "medium"), "deliver": ("hand over", "medium"), "pass": ("hand over", "medium"),
    "open": ("open", "high"), "close": ("close", "high"), "turn on": ("turn on", "high"), "turn off": ("turn off", "high"),
    "push": ("push", "high"), "stack": ("stack", "high"), "cut": ("cut", "high"), "pour": ("pour", "high"),
    "smash": ("smash", "high"), "stab": ("stab", "high"), "flip": ("flip", "high"), "hold": ("hold", "high"),
}
VERB_RX = re.compile(r"\b(pick up|pick|grab|grasp|get|retrieve|put|place|move|bring|deliver|pass|open|close|turn on|turn off|push|stack|cut|pour|smash|stab|flip|hold)\b")
# asset name -> words used for it in instructions (hand-curated, checked against the task texts)
ALIASES = {"wood_bowl": ["wooden bowl"], "eyeglasses_1": ["glasses"], "akita_black_bowl": ["akita black bowl", "black bowl"],
           "white_yellow_mug": ["white yellow mug", "yellow and white mug"], "can_of_soda__6": ["can of soda", "soda can"],
           "can_of_soda_3_with_hand": ["soda can"], "plate_with_hand": ["plate in my hand"], "left_hand": ["hand", "my hand"],
           "chefmate_8_frypan": ["frypan"], "frypan": ["frypan"], "salad_porcelain_plate": ["porcelain plate"],
           "cream_cheese": ["cream cheese box"], "cookies": ["cookies box"], "apple__7": ["red apple"], "porcelain_mug": ["porcelain mug", "white mug"]}


def blocks(text, key):
    i = text.find(f"(:{key}")
    if i < 0: return ""
    d, j = 0, i
    while True:
        d += text[j] == "("; d -= text[j] == ")"; j += 1
        if d == 0: return text[i:j]


def parse_bddl(path):
    t = path.read_text()
    return {"language": re.search(r"\(:language\s+([^)]*)\)", t).group(1).strip(),
            "problem": re.search(r"\(problem\s+(\w+)", t).group(1),
            "objects": [x[1] for x in re.findall(r"(\S+)\s+-\s+(\S+)", blocks(t, "objects"))],
            "fixtures": [x[1] for x in re.findall(r"(\S+)\s+-\s+(\S+)", blocks(t, "fixtures"))]}


def canonical(obj):
    """asset type -> (canonical key, flags). Rotated/held variants collapse onto the base object; this is a *normalization*, flagged."""
    flags = []
    k = obj
    if k.startswith("rotated_"): k = k[len("rotated_"):]; flags.append("rotated_variant")
    if k.endswith("_with_hand"): flags.append("held_by_hand")
    return k, flags


def main():
    src = (LIB / "benchmark/vla_safety_task_map.py").read_text()
    registry = ast.literal_eval(src.split("=", 1)[1].strip())
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    paper = (HERE.parent / "cache/refs/ls_paper.txt").read_text().splitlines() if (HERE.parent / "cache/refs/ls_paper.txt").exists() else []
    tasks, uncertain = [], []
    for suite, levels in registry.items():
        for lvl, names in levels.items():
            for name in names:
                sub = f"L{lvl}" if suite in SAFETY_SUITES else ""
                p = LIB / "bddl_files" / suite / sub / f"{name}.bddl"
                tid = f"{suite}/{sub + '/' if sub else ''}{name}"
                t = {"task_id": tid, "suite": suite, "level": f"L{lvl}" if suite in SAFETY_SUITES else None, "registry_name": name,
                     "registry_text": name.split("SCENE")[-1].lstrip("0123456789_").replace("_", " ") if "SCENE" in name else name.replace("_", " ")}
                if p.exists():
                    t.update(bddl=str(p.relative_to(REPO)), **parse_bddl(p))
                else:
                    t.update(bddl=None, language=t["registry_text"], problem=None, objects=[], fixtures=[])
                    uncertain.append({"item": tid, "issue": "registry name has no BDDL file (name truncated on disk); text taken from the registry name only"})
                if t["language"].lower().replace(" ", "") != t["registry_text"].lower().replace(" ", ""):
                    t["language_differs_from_registry_name"] = True
                tasks.append(t)

    verbs = defaultdict(lambda: {"sources": []})
    for t in tasks:
        for m in VERB_RX.finditer(t["language"].lower()):
            v = m.group(1)
            if v == "get" and "get" in t["language"].lower().split()[1:]: continue
            verbs[v]["sources"].append({"task_id": t["task_id"], "text": t["language"]})
    verb_out = {}
    for v, e in sorted(verbs.items()):
        g, conf = VERB_GROUPS[v]
        verb_out[v] = {"normalized_group": g, "normalization_confidence": conf, "n_tasks": len({s["task_id"] for s in e["sources"]}),
                       "sources": e["sources"][:6], "suites": sorted({s["task_id"].split("/")[0] for s in e["sources"]}),
                       "extraction": "regex over the BDDL :language text of registry tasks"}
        if conf == "low": uncertain.append({"item": f"verb:{v}", "issue": "normalization low confidence (appears only in a paraphrased human_safety L2 text)"})

    objects = defaultdict(lambda: {"tasks": [], "flags": set()})
    for t in tasks:
        for o in t["objects"] + t["fixtures"]:
            k, fl = canonical(o)
            e = objects[k]; e["asset_types"] = sorted(set(e.get("asset_types", [])) | {o}); e["flags"] |= set(fl)
            e["tasks"].append({"task_id": t["task_id"], "bddl": t["bddl"], "role": "fixture" if o in t["fixtures"] else "object"})
    object_out = {}
    for k, e in sorted(objects.items()):
        named = [t["task_id"] for t in tasks for a in ALIASES.get(k, []) + [k.replace("_", " ")] if a in t["language"].lower()]
        object_out[k] = {"asset_types": e["asset_types"], "flags": sorted(e["flags"]), "instruction_aliases": ALIASES.get(k, []),
                         "n_tasks": len({x["task_id"] for x in e["tasks"]}), "suites": sorted({x["task_id"].split("/")[0] for x in e["tasks"]}),
                         "named_in_instruction_of": sorted(set(named))[:6], "sources": e["tasks"][:5]}
        if re.search(r"__\d+$|_\d+$", k) and not k.endswith("_with_hand"):
            uncertain.append({"item": f"object:{k}", "issue": "numbered asset variant; the base category name is a normalization (e.g. apple__1 vs apple__7)"})

    scenes = {}
    for t in tasks:
        table = next((f for f in t["fixtures"] if f.endswith("table") or f == "floor"), None)
        scenes[t["task_id"]] = {"bddl": t["bddl"], "problem": t["problem"], "table": table, "fixtures": t["fixtures"], "objects": t["objects"],
                                "suite": t["suite"], "level": t["level"],
                                "has_human_hand": any("hand" in o for o in t["objects"]),
                                "visual_asset": {"image": None, "status": "no rendered image in the repo; scene exists as BDDL (+ HF asset pack, not downloaded)"}}

    variant_objs = set()
    for f in (LIB / "bddl_files").glob("*/*.bddl"):
        if f.parent.name != "randomizer" and (VARIANT.search(f.name) or f.parent.name == "libero_mix"):
            variant_objs |= {x[1] for x in re.findall(r"(\S+)\s+-\s+(\S+)", blocks(f.read_text(), "objects"))}
    registry_objs = {o for t in tasks for o in t["objects"]}

    lvl_lines = [i for i, l in enumerate(paper) if "L0 (Explicit Harm)" in l or "L1 (Physical Common Sense" in l or "L2 (Contextual Traps)" in l]
    out = {
        "source": {"repo": "https://github.com/LIBERO-SAFETY/LIBERO-Safety", "commit": commit, "registry": "libero/libero/benchmark/vla_safety_task_map.py",
                   "paper": "https://arxiv.org/abs/2606.23686", "n_registry_tasks": len(tasks), "scope_note": __doc__.split("Scope = ")[1].split("\n\n")[0] if False else "registry tasks only"},
        "semantic_levels": {
            "source": "paper Appendix B 'Semantic Safety Reasoning Suite' (cache/refs/ls_paper.txt lines %s)" % ", ".join(str(i + 1) for i in lvl_lines),
            "L0": "Explicit Harm: textual prompts contain explicit directives intended to cause direct environmental damage.",
            "L1": "Physical Common Sense Violations: linguistic instructions conceal fundamental violations of physical common sense within seemingly benign requests.",
            "L2": "Contextual Traps: subtle contextual traps designed to induce unsafe physical interactions through implicit environmental dependencies.",
            "note": "Levels measure the reasoning needed to see the hazard, not its magnitude. Paper text for the SSR suite says its instructions were generated with Qwen3-8B."},
        "suites": {s: {"levels": sorted({t["level"] for t in tasks if t["suite"] == s and t["level"]}), "n_tasks": sum(t["suite"] == s for t in tasks)} for s in registry},
        "tasks": tasks, "verbs": verb_out, "objects": object_out, "scenes": scenes,
        "variant_only": {"note": "appear only in augmented BDDL variants (libero_mix, _add_N, copy), not in registry tasks; NOT usable for generation",
                         "objects": sorted(variant_objs - registry_objs)},
        "not_in_repo": {"note": "the paper cites 953 distinct objects / 462 hand-object pairs in the asset pack on HuggingFace; only objects referenced by BDDL files are counted here"},
        "uncertain": uncertain,
    }
    (HERE / "libero_safety_components.json").write_text(json.dumps(out, indent=1))
    print(len(tasks), "tasks;", len(verb_out), "verbs;", len(object_out), "objects;", len(scenes), "scenes;", len(uncertain), "uncertain")
    print("verbs:", {v: e["n_tasks"] for v, e in verb_out.items()})


if __name__ == "__main__":
    main()
