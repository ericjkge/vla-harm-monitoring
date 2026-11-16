"""Build the group review page: every pair with both scenes and instructions side by side.

Reads libero_safety/minimal_pairs_150.jsonl, scene_manifest.json, validation_results.jsonl and the PNGs in
cache/scene_images; writes results/scene_review/index.html with the JPEG thumbnails embedded (one self-contained file).
Scenes that failed the placement check are shown first.

Usage: .venv/bin/python libero_safety/build_review_report.py
"""
import json
from html import escape as h
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "scene_review"
CAMS = ["agentview", "wrist", "front"]


def scene_key(sc):  # same key as render_scenes.py
    import hashlib
    blob = json.dumps([sc["scene_id"], sc["additions"], sc["arrangement"]], sort_keys=True)
    return f"{sc['scene_id'].split('/')[-1][:48]}__{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def thumbs(man):
    """JPEG data URIs, one per scene view; the page stores each once and fills the <img> tags from this table."""
    import base64, io
    out = {}
    for k, m in man.items():
        for cam in CAMS:
            buf = io.BytesIO()
            Image.open(ROOT / m["images"][cam]).convert("RGB").save(buf, "JPEG", quality=80)
            out[f"{k}__{cam}"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


def fail_text(r):
    return (f"{r['object']} ended {r['xy_dist']} m from {r['target']} (limit 0.10 m), "
            f"{'below' if r['dz'] < 0 else 'above'} it by {abs(r['dz'])} m")


def arrangement_text(sc):
    bits = [f"added {a['object']} (from {a['source_task'].split('/')[-1].replace('_', ' ')})" for a in sc["additions"]]
    for a in sc["arrangement"]:
        tgt = a.get("in") or a.get("on")
        bits.append(f"{a['object']} {'in' if a.get('in') else 'on'} {tgt}" + (f" ({a['region']})" if a.get("region") else ""))
    return "; ".join(bits) or "benchmark scene unchanged"


def images_html(k, man, lazy=True):
    m = man[k]
    return "".join(
        f'<figure><img data-img="{k}__{c}" alt="{c} view" width="256" height="256">'
        f"<figcaption>{c}{' (QA only)' if c == 'front' else ''}</figcaption></figure>" for c in CAMS)


def side_html(p, side, man, other_key):
    sc = p[side]["scene"]; k = scene_key(sc); m = man[k]
    flags = []
    for r in m.get("placement_check", []):
        if not r["ok"]:
            flags.append(f'<p class="flag bad">Placement check failed: {h(fail_text(r))}</p>')
    if m.get("fell_off"):
        flags.append(f'<p class="flag warn">Fell off the table while settling: {h(", ".join(m["fell_off"]))}</p>')
    same = k == other_key and side == "harmful"
    return f"""<div class="side {side}">
  <div class="sidehead"><span class="tag">{side}</span><span class="inst">{h(p[side]['instruction'])}</span></div>
  <div class="imgs">{images_html(k, man)}</div>
  {'<p class="same">Same scene image as the harmless side; only the instruction changes.</p>' if same else ''}
  <dl class="meta"><dt>Scene</dt><dd>{h(sc['scene_id'])}</dd>
  <dt>Setup</dt><dd>{h(arrangement_text(sc))}</dd>
  {''.join(f'<dt>Note</dt><dd>{h(n)}</dd>' for n in m.get('notes', []))}</dl>
  {''.join(flags)}
</div>"""


def main():
    pairs = [json.loads(l) for l in open(ROOT / "libero_safety/minimal_pairs_150.jsonl")]
    man = json.load(open(ROOT / "libero_safety/scene_manifest.json"))
    val = {json.loads(l)["pair_id"]: json.loads(l) for l in open(ROOT / "libero_safety/validation_results.jsonl")}
    imgs = thumbs(man)

    failed = {k: m for k, m in man.items() if any(not r["ok"] for r in m.get("placement_check", []))}
    fell = {k for k, m in man.items() if m.get("fell_off")}
    uses = {}
    for p in pairs:
        for side in ("harmless", "harmful"):
            uses.setdefault(scene_key(p[side]["scene"]), []).append((p, side))

    # --- failed-scene section
    fcards = []
    for k, m in failed.items():
        rows = "".join(f'<li><a href="#{p["pair_id"]}">{p["pair_id"]}</a> {side}: {h(p[side]["instruction"])}</li>' for p, side in uses[k])
        why = "".join(f"<li>{h(fail_text(r))}</li>" for r in m["placement_check"] if not r["ok"])
        fcards.append(f"""<article class="failcard">
  <h3>{h(m['scene_id'].split('/')[-1].replace('_', ' '))}</h3>
  <p class="id">{h(k)}</p>
  <div class="imgs">{images_html(k, man, lazy=False)}</div>
  <p class="label">What went wrong</p><ul>{why}</ul>
  <p class="label">Pairs that use this scene</p><ul>{rows}</ul>
</article>""")

    # --- pair cards
    order = {"scene": 0, "object": 1, "verb": 2}
    sections = []
    for comp in ("scene", "object", "verb"):
        cards = []
        for p in [q for q in pairs if q["changed_component"] == comp]:
            ks = {scene_key(p[s]["scene"]) for s in ("harmless", "harmful")}
            bad = bool(ks & set(failed)); wf = bool(ks & fell)
            v = val[p["pair_id"]]["status"]
            badges = (f'<span class="b fail">scene failed placement</span>' if bad else "") + (f'<span class="b warnb">object fell off</span>' if wf else "")
            hk = scene_key(p["harmless"]["scene"])
            cards.append(f"""<section class="pair" id="{p['pair_id']}" data-comp="{comp}" data-harm="{p['harm_type']}" data-val="{v}" data-flag="{1 if (bad or wf) else 0}" data-bad="{1 if bad else 0}">
  <header><h3>{p['pair_id']}</h3>
  <span class="b">{p['harm_type'].replace('_', ' ')}</span><span class="b">{p['semantic_level']} ({p['semantic_level_confidence']} confidence)</span>
  <span class="b {'ok' if v == 'valid' else 'unc'}">auto-check: {v}</span>{badges}</header>
  <p class="mech"><strong>Why it is harmful:</strong> {h(p['harm_mechanism'])} <span class="fam">hazard family: {h(p['hazard_family'])}</span></p>
  <div class="cols">{side_html(p, 'harmless', man, hk)}{side_html(p, 'harmful', man, hk)}</div>
</section>""")
        sections.append(f'<h2 id="{comp}">Changed component: {comp} <small>({len(cards)} pairs)</small></h2>' + "".join(cards))

    n_ok = sum(1 for v in val.values() if v["status"] == "valid")
    html = PAGE.format(
        npairs=len(pairs), nscenes=len(man), nfail=len(failed), nfell=len(fell), nvalid=n_ok, nunc=len(pairs) - n_ok,
        failcards="".join(fcards), sections="".join(sections))
    html += "<script>\n(function(){var IMG=" + json.dumps(imgs) + ";document.querySelectorAll('img[data-img]').forEach(function(i){i.src=IMG[i.dataset.img];});})();\n</script>\n"
    (OUT / "index.html").write_text(html)
    print(f"{len(pairs)} pairs, {len(man)} scenes, {len(failed)} failed, {len(fell)} with fallen objects -> {OUT/'index.html'}")


PAGE = """<title>Scene Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: one long review sheet; failed scenes first, then pairs grouped by the component that changes */
:root {{
  --bg:#f6f6f4; --surface:#ffffff; --fg:#1d2127; --muted:#5d6670; --line:#dcdfe2;
  --harmless:#1f6f8b; --harmless-bg:#e8f2f6; --harmful:#b4411f; --harmful-bg:#fbece6;
  --fail:#b3261e; --fail-bg:#fdeceb; --warn:#8a5a00; --warn-bg:#fff4dc; --ok:#2b6b43; --ok-bg:#e6f3ea;
  --font:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#14171a; --surface:#1c2024; --fg:#e8eaec; --muted:#9aa3ab; --line:#32383e;
  --harmless:#6fbcd6; --harmless-bg:#16303a; --harmful:#f0916f; --harmful-bg:#3a2119;
  --fail:#ff8a80; --fail-bg:#3d1c1a; --warn:#f0c36a; --warn-bg:#392d12; --ok:#7fcf9b; --ok-bg:#163223; color-scheme:dark; }} }}
:root[data-theme="dark"] {{
  --bg:#14171a; --surface:#1c2024; --fg:#e8eaec; --muted:#9aa3ab; --line:#32383e;
  --harmless:#6fbcd6; --harmless-bg:#16303a; --harmful:#f0916f; --harmful-bg:#3a2119;
  --fail:#ff8a80; --fail-bg:#3d1c1a; --warn:#f0c36a; --warn-bg:#392d12; --ok:#7fcf9b; --ok-bg:#163223; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--font); font-size:15px; line-height:1.5;
  padding-inline:max(16px, env(safe-area-inset-left, 0px)) max(16px, env(safe-area-inset-right, 0px)); padding-block:24px 64px; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:28px; margin:0 0 4px; text-wrap:balance; }}
h2 {{ font-size:20px; margin:40px 0 14px; text-wrap:balance; }} h2 small {{ color:var(--muted); font-weight:400; }}
h3 {{ font-size:15px; margin:0; font-family:var(--mono); font-weight:500; }}
.lede {{ color:var(--muted); max-width:70ch; margin:0 0 18px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:8px 22px; margin:0 0 18px; padding:0; list-style:none; color:var(--muted); }}
.stats b {{ color:var(--fg); font-variant-numeric:tabular-nums; }}
.howto {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin-bottom:8px; }}
.howto p {{ margin:4px 0; max-width:80ch; }}
.failbox {{ border:2px solid var(--fail); background:var(--fail-bg); border-radius:10px; padding:16px; margin:28px 0 8px; }}
.failbox h2 {{ margin:0 0 6px; color:var(--fail); }}
.failbox > p {{ margin:0 0 14px; max-width:78ch; }}
.failgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:14px; }}
.failcard {{ background:var(--surface); border:1px solid var(--fail); border-radius:8px; padding:12px 14px; min-width:0; }}
.failcard h3 {{ font-family:var(--font); font-size:15px; font-weight:600; text-transform:capitalize; }}
.id {{ font-family:var(--mono); font-size:11px; color:var(--muted); margin:2px 0 8px; word-break:break-all; }}
.label {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:10px 0 2px; }}
.failcard ul {{ margin:0; padding-left:18px; }} .failcard li {{ margin:2px 0; }}
.filters {{ position:sticky; top:env(safe-area-inset-top, 0px); z-index:5; background:var(--bg); padding:10px 0; display:flex; flex-wrap:wrap; gap:6px 18px; border-bottom:1px solid var(--line); margin-top:22px; }}
.filters fieldset {{ border:0; padding:0; margin:0; display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
.filters legend {{ float:left; margin-right:6px; color:var(--muted); font-size:12px; }}
.filters button {{ font:inherit; font-size:13px; border:1px solid var(--line); background:var(--surface); color:var(--fg); padding:4px 10px; border-radius:6px; cursor:pointer; }}
.filters button[aria-pressed="true"] {{ background:var(--fg); color:var(--bg); border-color:var(--fg); }}
.filters button:focus-visible, a:focus-visible {{ outline:2px solid var(--harmless); outline-offset:2px; }}
.count {{ margin-left:auto; color:var(--muted); font-size:13px; align-self:center; }}
.pair {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px; margin:0 0 16px; scroll-margin-top:70px; }}
.pair header {{ display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; }}
.b {{ font-size:12px; border:1px solid var(--line); border-radius:999px; padding:1px 9px; color:var(--muted); white-space:nowrap; }}
.b.ok {{ color:var(--ok); background:var(--ok-bg); border-color:transparent; }} .b.unc {{ color:var(--warn); background:var(--warn-bg); border-color:transparent; }}
.b.fail {{ color:var(--fail); background:var(--fail-bg); border-color:var(--fail); font-weight:600; }} .b.warnb {{ color:var(--warn); background:var(--warn-bg); border-color:transparent; }}
.mech {{ margin:8px 0 12px; max-width:90ch; }} .fam {{ color:var(--muted); font-size:12px; margin-left:6px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
@media (max-width:860px) {{ .cols {{ grid-template-columns:1fr; }} }}
.side {{ border-radius:8px; padding:10px; min-width:0; }}
.side.harmless {{ background:var(--harmless-bg); }} .side.harmful {{ background:var(--harmful-bg); }}
.sidehead {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 10px; margin-bottom:8px; }}
.tag {{ font-size:11px; letter-spacing:.07em; text-transform:uppercase; font-weight:600; }}
.harmless .tag {{ color:var(--harmless); }} .harmful .tag {{ color:var(--harmful); }}
.inst {{ font-size:17px; font-weight:600; text-wrap:balance; }}
.imgs {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
figure {{ margin:0; min-width:0; }} figure img {{ display:block; width:100%; height:auto; border-radius:4px; background:#000; }}
figcaption {{ font-size:11px; color:var(--muted); margin-top:2px; }}
.same {{ font-size:12px; color:var(--muted); margin:6px 0 0; font-style:italic; }}
.meta {{ display:grid; grid-template-columns:auto 1fr; gap:2px 10px; font-size:12.5px; margin:8px 0 0; }}
.meta dt {{ color:var(--muted); }} .meta dd {{ margin:0; min-width:0; overflow-wrap:anywhere; }}
.flag {{ font-size:12.5px; border-radius:6px; padding:5px 8px; margin:8px 0 0; }}
.flag.bad {{ background:var(--fail-bg); color:var(--fail); border:1px solid var(--fail); }} .flag.warn {{ background:var(--warn-bg); color:var(--warn); }}
a {{ color:var(--harmless); }}
[hidden] {{ display:none!important; }}
</style>
<main>
<h1>Scene review: LIBERO-Safety pairs</h1>
<p class="lede">Each card is one minimal pair: the harmless and harmful instruction, each with the scene it is paired with, as rendered from the benchmark simulator. Please check that the harm label is right, that the scene shows what the setup line says, and that the two sides differ only in the one component named.</p>
<ul class="stats"><li><b>{npairs}</b> pairs</li><li><b>{nscenes}</b> unique scenes</li><li><b>{nvalid}</b> auto-check valid, <b>{nunc}</b> uncertain</li><li><b style="color:var(--fail)">{nfail}</b> scenes failed placement</li><li><b>{nfell}</b> scenes with a fallen object</li></ul>
<div class="howto">
<p><strong>Views.</strong> agentview and wrist are the two cameras the robot policy sees. front is an extra angle for reviewers only.</p>
<p><strong>Verb pairs</strong> use one scene on both sides, so the images are the same. The harm is only in the words.</p>
<p><strong>Auto-check: uncertain</strong> means the script's lexicon labelled it with medium confidence. It does not mean wrong.</p>
</div>

<div class="failbox" id="failed">
<h2>{nfail} scenes failed the placement check</h2>
<p>In each of these composed scenes an object did not end up where the setup line says it should, so the picture does not show the intended hazard. Pairs that use them should be fixed or dropped before activations are extracted.</p>
<div class="failgrid">{failcards}</div>
</div>

<nav class="filters" aria-label="Filters">
<fieldset><legend>Changed</legend><button type="button" data-f="comp" data-v="all" aria-pressed="true">all</button><button type="button" data-f="comp" data-v="scene" aria-pressed="false">scene</button><button type="button" data-f="comp" data-v="object" aria-pressed="false">object</button><button type="button" data-f="comp" data-v="verb" aria-pressed="false">verb</button></fieldset>
<fieldset><legend>Harm</legend><button type="button" data-f="harm" data-v="all" aria-pressed="true">all</button><button type="button" data-f="harm" data-v="explicit_harm" aria-pressed="false">explicit</button><button type="button" data-f="harm" data-v="implied_harm" aria-pressed="false">implied</button></fieldset>
<fieldset><legend>Show</legend><button type="button" data-f="flag" data-v="all" aria-pressed="true">all</button><button type="button" data-f="flag" data-v="bad" aria-pressed="false">failed scenes</button><button type="button" data-f="flag" data-v="any" aria-pressed="false">any scene flag</button></fieldset>
<span class="count" id="count"></span>
</nav>
{sections}
</main>
<script>
(function () {{
  var sel = {{comp: 'all', harm: 'all', flag: 'all'}};
  var pairs = Array.prototype.slice.call(document.querySelectorAll('.pair'));
  function apply() {{
    var n = 0;
    pairs.forEach(function (p) {{
      var ok = (sel.comp === 'all' || p.dataset.comp === sel.comp) && (sel.harm === 'all' || p.dataset.harm === sel.harm) &&
        (sel.flag === 'all' || (sel.flag === 'bad' ? p.dataset.bad === '1' : p.dataset.flag === '1'));
      p.hidden = !ok; if (ok) n++;
    }});
    document.querySelectorAll('h2[id]').forEach(function (h) {{ if (h.id !== 'failed') {{ var any = pairs.some(function (p) {{ return p.dataset.comp === h.id && !p.hidden; }}); h.hidden = !any; }} }});
    document.getElementById('count').textContent = n + ' of ' + pairs.length + ' pairs shown';
  }}
  document.querySelectorAll('.filters button').forEach(function (b) {{
    b.addEventListener('click', function () {{
      sel[b.dataset.f] = b.dataset.v;
      document.querySelectorAll('.filters button[data-f="' + b.dataset.f + '"]').forEach(function (x) {{ x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); }});
      apply();
    }});
  }});
  apply();
}})();
</script>
"""

if __name__ == "__main__":
    main()
