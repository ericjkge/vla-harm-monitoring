"""Extract every LIBERO task (suite, scene, language instruction, objects, fixtures) from the BDDL files.

Usage: python scripts/pull_libero_prompts.py [--libero cache/LIBERO] [--out results/libero_prompts.jsonl]
Clone first: git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git cache/LIBERO
"""
import argparse, json, re
from pathlib import Path


def sexp_block(text, key):
    """Return the inner text of the first '(:key ...)' block, with balanced parens."""
    i = text.find(f"(:{key}")
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(text):
        depth += text[j] == "("
        depth -= text[j] == ")"
        j += 1
        if depth == 0:
            break
    return text[i + len(key) + 2 : j - 1]


def parse(path):
    text = path.read_text()
    lang = re.search(r"\(:language\s+([^)]*)\)", text).group(1).strip()
    objs = re.findall(r"(\S+)\s+-\s+(\S+)", sexp_block(text, "objects"))
    fixt = re.findall(r"(\S+)\s+-\s+(\S+)", sexp_block(text, "fixtures"))
    return {
        "suite": path.parent.name,
        "file": path.stem,
        "scene": scene_name(path.stem) if "_SCENE" in path.stem else path.parent.name,
        "language": lang,
        "objects": sorted({t for _, t in objs}),
        "fixtures": sorted({t for _, t in fixt if t not in ("floor", "main_table")}),
        "table": next((t for _, t in fixt if t.endswith("_table")), None),
    }


def scene_name(stem):
    m = re.match(r"((?:[A-Z_0-9]+?)_SCENE\d+)", stem)
    return m.group(1) if m else stem


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", default="cache/LIBERO")
    ap.add_argument("--out", default="results/libero_prompts.jsonl")
    a = ap.parse_args()
    root = Path(a.libero) / "libero/libero/bddl_files"
    rows = [parse(p) for p in sorted(root.glob("*/*.bddl"))]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} tasks -> {a.out}")
