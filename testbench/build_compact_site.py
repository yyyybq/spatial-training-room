"""
build_compact_site.py - Generate a lightweight GitHub Pages review site.

Combines project introduction (hero, stats, template catalogue, validation)
with rendered task examples (init/target/topdown images, Q&A, actions).
Total output is typically 4-8 MB, suitable for GitHub Pages.

Usage:
    python testbench/build_compact_site.py \
        --rendered-root out/fixed_v6_rendered \
        --jsonl-root    out/fixed_v6 \
        --out           ../spatial-training-room-page/index.html
"""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

# Template metadata.

TMPL_META = {
    "T01": ("Appearance Discrimination",   "occlusion",   0.70,  8),
    "T04": ("Actual Size Comparison",       "spatial",     0.90,  3),
    "T05": ("Actual Distance Comparison",   "spatial",     0.66,  9),
    "T06": ("Passage Width Assessment",     "navigation",  0.74,  7),
    "T08": ("Configuration / Arrangement",  "spatial",     0.66,  9),
    "T11": ("Single vs. Multi Instance",    "counting",    0.95,  2),
    "T13": ("Occlusion to Category",        "occlusion",   0.86,  4),
    "T17": ("Occluded Object Existence",    "occlusion",   0.95,  2),
    "T18": ("Occluded Object Category",     "occlusion",   0.63, 10),
    "T19": ("Occluded Object Completeness", "occlusion",   0.77,  6),
    "T20": ("Local Direction Search",       "navigation",  0.90,  3),
    "T21": ("Nearest-Neighbor Target",      "spatial",     0.60, 11),
    "T23": ("Cross-Room Existence",         "cross-room",  0.86,  4),
    "T24": ("Doorway Direction",            "navigation",  0.74,  7),
    "T26": ("Occlusion Counting",           "counting",    0.86,  4),
    "T27": ("Zone Counting",                "counting",    0.60, 11),
    "T29": ("Contact Relationship",         "spatial",     0.90,  3),
    "T32": ("Connectivity Judgment",        "navigation",  0.95,  2),
    "T33": ("Passage Traversability",       "navigation",  0.70,  8),
}

CATEGORY_COLOR = {
    "occlusion":   ("#eff6ff", "#1e40af", "#bfdbfe"),
    "spatial":     ("#fefce8", "#92400e", "#fde68a"),
    "counting":    ("#f0fdf4", "#166534", "#86efac"),
    "navigation":  ("#fdf4ff", "#6b21a8", "#e9d5ff"),
    "cross-room":  ("#fff7ed", "#9a3412", "#fed7aa"),
}

# Helpers.

def img_to_b64(path: Path, max_w: int = 480, quality: int = 65) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#16a34a" if score >= 0.8 else "#d97706" if score >= 0.6 else "#dc2626"
    return (f'<div class="sbar-wrap">'
            f'<div class="sbar" style="width:{pct}px;background:{color}"></div>'
            f'<span>{score:.2f}</span></div>')


# Main.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-root", required=True)
    ap.add_argument("--jsonl-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-label", default="dataset")
    ap.add_argument("--max-w", type=int, default=480)
    ap.add_argument("--quality", type=int, default=65)
    args = ap.parse_args()

    rendered_root = Path(args.rendered_root)
    jsonl_root = Path(args.jsonl_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmpl_dirs = sorted(
        d for d in rendered_root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )

    sections_html = []
    catalogue_rows = []
    total_images = 0
    total_tasks = 0
    score_values = []
    step_values = []

    for tmpl_dir in tmpl_dirs:
        tmpl_id = tmpl_dir.name
        manifest = json.loads((tmpl_dir / "manifest.json").read_text(encoding="utf-8"))
        tasks_meta = {t["task_dir"]: t for t in manifest.get("tasks", [])}

        jsonl_rows = load_jsonl(jsonl_root / f"{tmpl_id}.jsonl")
        jsonl_by_idx = {i: r for i, r in enumerate(jsonl_rows)}

        task_dirs = sorted(d for d in tmpl_dir.iterdir() if d.is_dir())
        if not task_dirs:
            continue

        desc, cat, avg_score, avg_steps = TMPL_META.get(
            tmpl_id, (tmpl_id, "other", 0.0, 0))
        bg, fg, border = CATEGORY_COLOR.get(cat, ("#f5f5f5", "#333", "#ccc"))
        rendered_count = len(task_dirs)
        total_tasks += rendered_count
        tmpl_scores = [
            float(r.get("score", 0.0))
            for r in jsonl_rows[:rendered_count]
            if r.get("score") is not None
        ]
        tmpl_steps = [
            len(r.get("action_sequence") or [])
            for r in jsonl_rows[:rendered_count]
        ]
        if tmpl_scores:
            avg_score = sum(tmpl_scores) / len(tmpl_scores)
            score_values.extend(tmpl_scores)
        if tmpl_steps:
            avg_steps = sum(tmpl_steps) / len(tmpl_steps)
            step_values.extend(tmpl_steps)

        # catalogue row
        catalogue_rows.append(
            f'<tr>'
            f'<td><a href="#{tmpl_id}" class="tmpl-link">{tmpl_id}</a></td>'
            f'<td>{desc}</td>'
            f'<td><span class="cat-tag" style="background:{bg};color:{fg};border-color:{border}">{cat}</span></td>'
            f'<td>{rendered_count}</td>'
            f'<td>{score_bar(avg_score)}</td>'
            f'<td>{avg_steps:.1f}</td>'
            f'</tr>'
        )

        task_cards = []
        for task_dir in task_dirs:
            meta = tasks_meta.get(task_dir.name, {})
            idx = meta.get("task_index", 0)
            row = jsonl_by_idx.get(idx, {})

            question = row.get("question", row.get("q", "N/A"))
            answer   = row.get("gt_answer", row.get("answer", "N/A"))
            choices  = row.get("choices", [])
            actions  = meta.get("action_descriptions", [])
            n_frames = meta.get("num_frames", "?")
            task_id_str = meta.get("task_id", task_dir.name)

            imgs = {}
            for key, fname in [("init", "init.png"), ("target", "target.png"), ("topdown", "topdown.png")]:
                p = task_dir / fname
                if p.exists():
                    imgs[key] = img_to_b64(p, args.max_w, args.quality)
                    total_images += 1

            choices_html = ""
            if choices:
                items = "".join(f'<li>{c}</li>' for c in choices)
                choices_html = f'<ul class="choices">{items}</ul>'

            actions_html = ""
            if actions:
                acts = "".join(f'<li>{a}</li>' for a in actions)
                actions_html = (f'<details class="actions-detail">'
                                f'<summary>Expert actions &nbsp;({n_frames} steps)</summary>'
                                f'<ol>{acts}</ol></details>')

            def img_tag(key, label):
                if key not in imgs:
                    return ""
                return (f'<div class="frame-cell">'
                        f'<div class="frame-label">{label}</div>'
                        f'<img src="{imgs[key]}" loading="lazy"/></div>')

            task_cards.append(f"""
<div class="task-card">
  <div class="task-header">
    <span class="task-id">{task_id_str}</span>
    <span class="task-steps">{n_frames} steps</span>
  </div>
  <div class="qa-row">
    <div class="question"><span class="qlabel">Q</span>{question}</div>
    {choices_html}
    <div class="answer"><span class="alabel">A</span><span class="ans-text">{answer}</span></div>
  </div>
  <div class="frames-row">
    {img_tag("init",    "1. Init view - unanswerable")}
    {img_tag("topdown", "2. Top-down navigation path")}
    {img_tag("target",  "3. Target view - answerable")}
  </div>
  {actions_html}
</div>""")

        sections_html.append(f"""
<section class="tmpl-section" id="{tmpl_id}">
  <div class="tmpl-section-header">
    <span class="tmpl-badge">{tmpl_id}</span>
    <div class="tmpl-info">
      <span class="tmpl-desc">{desc}</span>
      <span class="cat-tag" style="background:{bg};color:{fg};border-color:{border}">{cat}</span>
    </div>
    <div class="tmpl-stats">
      <span>score {avg_score:.2f}</span>
      <span>avg {avg_steps} steps</span>
      <span>{len(task_dirs)} tasks</span>
    </div>
  </div>
  {''.join(task_cards)}
</section>""")

    print(f"[info] {len(sections_html)} templates, {total_tasks} tasks, {total_images} images")

    overall_score = sum(score_values) / len(score_values) if score_values else 0.0
    overall_steps = sum(step_values) / len(step_values) if step_values else 0.0

    nav_links = "".join(
        f'<a href="#{d.name}">{d.name} <span class="nav-desc">{TMPL_META.get(d.name, ("",))[0]}</span></a>'
        for d in tmpl_dirs
    )
    cat_table_rows = "".join(catalogue_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Spatial Training Room - APL Data Factory</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#fff;--bg2:#f7f7f7;--border:#e4e4e4;
  --text:#1a1a1a;--text2:#555;
  --accent:#2563eb;--green:#16a34a;
  --nav-w:210px;
}}
body{{font-family:"Segoe UI",system-ui,sans-serif;color:var(--text);background:var(--bg);line-height:1.6;}}

/* top bar */
.topbar{{
  position:sticky;top:0;z-index:200;
  background:rgba(255,255,255,.93);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--border);
  padding:0 20px;height:48px;
  display:flex;align-items:center;gap:16px;
}}
.topbar h1{{font-size:.95em;font-weight:700}}
.topbar .sub{{font-size:.78em;color:var(--text2);margin-left:auto}}
.topbar a{{font-size:.8em;color:var(--accent);text-decoration:none;}}

/* layout */
.layout{{display:flex;min-height:calc(100vh - 48px)}}
.sidebar{{
  width:var(--nav-w);flex-shrink:0;
  padding:12px 6px 16px;
  border-right:1px solid var(--border);
  position:sticky;top:48px;height:calc(100vh - 48px);
  overflow-y:auto;
}}
.sidebar .sec-label{{font-size:.68em;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:.05em;padding:8px 10px 4px;}}
.sidebar a{{
  display:block;padding:5px 10px;border-radius:5px;
  font-size:.8em;color:var(--text2);text-decoration:none;
  margin-bottom:1px;line-height:1.3;
}}
.sidebar a:hover{{background:var(--bg2);color:var(--accent)}}
.nav-desc{{display:block;font-size:.72em;color:#aaa;font-weight:400;}}

.main{{flex:1;padding:32px 36px 80px;max-width:960px;min-width:0;}}

/* hero */
.hero{{
  background:linear-gradient(135deg,#0f172a,#1e3a5f 60%,#1e40af);
  color:white;padding:40px 36px 32px;margin:-32px -36px 40px;
}}
.hero h1{{font-size:clamp(1.4em,3vw,2em);font-weight:800;line-height:1.2;margin-bottom:8px;}}
.hero .sub{{color:#93c5fd;font-size:.95em;margin-bottom:20px;max-width:620px;}}
.hero-links{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px;}}
.hero-links a{{
  padding:6px 16px;border-radius:6px;font-size:.82em;font-weight:600;
  text-decoration:none;transition:opacity .15s;
}}
.hero-links a:hover{{opacity:.85}}
.btn-w{{background:white;color:#1e40af;}}
.btn-o{{background:transparent;color:white;border:1.5px solid rgba(255,255,255,.45);}}
.hero-stats{{
  display:flex;gap:24px;flex-wrap:wrap;padding-top:24px;
  border-top:1px solid rgba(255,255,255,.15);
}}
.hstat .n{{font-size:1.8em;font-weight:800;color:#93c5fd;line-height:1;}}
.hstat .l{{font-size:.7em;color:#cbd5e1;margin-top:2px;}}

/* section headings */
.section-h{{
  font-size:1.05em;font-weight:700;
  display:flex;align-items:center;gap:8px;
  margin:40px 0 14px;
}}
.section-h::after{{content:'';flex:1;height:1px;background:var(--border);}}

/* abstract */
.abstract{{
  background:var(--bg2);border-left:4px solid var(--accent);
  padding:16px 20px;border-radius:0 8px 8px 0;
  font-size:.9em;color:var(--text2);line-height:1.75;
  margin-bottom:24px;
}}
.abstract code{{background:#e0e7ff;padding:1px 5px;border-radius:3px;color:#3730a3;font-size:.92em;}}

/* arch block */
.arch{{
  background:#0f172a;color:#e2e8f0;
  border-radius:8px;padding:20px 24px;
  font-family:"Cascadia Code","Consolas",monospace;
  font-size:.78em;line-height:1.9;overflow-x:auto;
  margin-bottom:24px;
}}
.arch .c{{color:#64748b;}} .arch .k{{color:#93c5fd;}} .arch .ar{{color:#34d399;}}

/* validation */
.val-grid{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;}}
.v-card{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:8px;padding:12px 14px;min-width:140px;text-align:center;
}}
.v-card .vt{{font-weight:700;font-size:.95em;}}
.v-card .vc{{font-size:.7em;color:var(--text2);margin:3px 0;min-height:2.4em;}}
.v-card .vr{{font-size:.75em;color:var(--green);font-weight:600;}}
.val-sum{{
  background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
  padding:14px 18px;display:flex;align-items:center;gap:16px;margin-bottom:8px;
}}
.val-sum .big{{font-size:2em;font-weight:800;color:var(--green);flex-shrink:0;}}
.val-sum p{{font-size:.85em;color:#166534;line-height:1.5;}}
.val-sum code{{background:#dcfce7;padding:1px 5px;border-radius:3px;}}

/* catalogue table */
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:8px;margin-bottom:8px;}}
table{{width:100%;border-collapse:collapse;font-size:.83em;}}
thead th{{background:var(--bg2);padding:9px 12px;text-align:left;font-weight:600;white-space:nowrap;}}
tbody td{{padding:8px 12px;border-top:1px solid var(--border);}}
tbody tr:hover td{{background:#fafafa;}}
.tmpl-link{{color:var(--accent);text-decoration:none;font-weight:600;}}
.tmpl-link:hover{{text-decoration:underline;}}
.cat-tag{{
  display:inline-block;padding:1px 7px;border-radius:4px;
  font-size:.75em;font-weight:600;border:1px solid;
}}
.sbar-wrap{{display:flex;align-items:center;gap:6px;}}
.sbar{{height:5px;border-radius:3px;}}

/* fix callout */
.fix-box{{border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:24px;}}
.fix-hdr{{background:var(--bg2);padding:10px 16px;font-weight:600;font-size:.83em;border-bottom:1px solid var(--border);}}
.fix-body{{padding:14px 16px;font-size:.85em;color:var(--text2);line-height:1.7;}}
.fix-body code{{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:.92em;}}

/* template sections */
.tmpl-section{{margin-bottom:52px;}}
.tmpl-section-header{{
  display:flex;align-items:center;flex-wrap:wrap;gap:10px;
  padding:10px 14px;background:var(--bg2);
  border:1px solid var(--border);border-radius:8px 8px 0 0;
  margin-bottom:0;
}}
.tmpl-badge{{
  background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;
  border-radius:6px;padding:2px 10px;font-size:.88em;font-weight:700;
  flex-shrink:0;
}}
.tmpl-info{{flex:1;min-width:0;}}
.tmpl-desc{{font-weight:600;font-size:.9em;display:block;}}
.tmpl-stats{{display:flex;gap:12px;font-size:.75em;color:var(--text2);flex-wrap:wrap;}}

/* task card */
.task-card{{
  border:1px solid var(--border);border-top:none;
  margin-bottom:0;overflow:hidden;
}}
.tmpl-section > .task-card:last-child{{border-radius:0 0 8px 8px;}}
.task-header{{
  background:#fafafa;padding:7px 14px;
  font-size:.75em;color:var(--text2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
}}
.task-id{{font-weight:600;color:var(--text);}}
.task-steps{{margin-left:auto;}}
.qa-row{{padding:10px 14px 6px;}}
.qlabel,.alabel{{
  display:inline-block;width:20px;height:20px;line-height:20px;
  text-align:center;border-radius:4px;font-size:.75em;font-weight:700;
  margin-right:6px;flex-shrink:0;
}}
.qlabel{{background:#eff6ff;color:#1e40af;}}
.alabel{{background:#f0fdf4;color:#166534;}}
.question{{font-size:.88em;margin-bottom:6px;line-height:1.5;display:flex;align-items:baseline;}}
.answer{{font-size:.88em;display:flex;align-items:baseline;}}
.ans-text{{
  display:inline-block;background:#dcfce7;color:#166534;
  border:1px solid #86efac;border-radius:4px;padding:1px 8px;font-weight:600;
}}
.choices{{padding:2px 14px 6px 40px;list-style:disc;}}
.choices li{{font-size:.8em;color:var(--text2);margin-bottom:1px;}}

.frames-row{{display:flex;border-top:1px solid var(--border);}}
.frame-cell{{flex:1;min-width:0;border-right:1px solid var(--border);}}
.frame-cell:last-child{{border-right:none;}}
.frame-label{{
  font-size:.68em;color:var(--text2);padding:3px 6px;
  background:var(--bg2);border-bottom:1px solid var(--border);text-align:center;
}}
.frame-cell img{{width:100%;display:block;}}

.actions-detail{{
  padding:7px 14px;border-top:1px solid var(--border);
  font-size:.78em;color:var(--text2);
}}
.actions-detail summary{{cursor:pointer;font-weight:600;color:var(--text);}}
.actions-detail ol{{margin:6px 0 3px 20px;line-height:1.7;}}

@media(max-width:700px){{
  .sidebar{{display:none;}}
  .main{{padding:16px 14px 60px;}}
  .hero{{margin:-16px -14px 24px;padding:28px 16px 24px;}}
  .frames-row{{flex-direction:column;}}
  .frame-cell{{border-right:none;border-bottom:1px solid var(--border);}}
  .frame-cell:last-child{{border-bottom:none;}}
}}
</style>
</head>
<body>

<div class="topbar">
  <h1>Spatial Training Room</h1>
  <span class="sub">{args.dataset_label} &nbsp;&middot;&nbsp; {len(sections_html)} templates &nbsp;&middot;&nbsp; {total_tasks} tasks &nbsp;&middot;&nbsp; {total_images} images</span>
</div>

<div class="layout">
  <nav class="sidebar">
    <div class="sec-label">Overview</div>
    <a href="#intro">Introduction</a>
    <a href="#validation">Validation</a>
    <a href="#catalogue">Template Catalogue</a>
    <div class="sec-label" style="margin-top:8px">Templates</div>
    {nav_links}
  </nav>

  <div class="main">

    <!-- Hero -->
    <div class="hero">
      <h1>Spatial Training Room</h1>
      <p class="sub">An Automated APL Data Factory for Active Spatial Perception in 3D Indoor Scenes</p>

      <div class="hero-stats">
        <div class="hstat"><div class="n">{len(sections_html)}</div><div class="l">Active Templates</div></div>
        <div class="hstat"><div class="n">{total_tasks}</div><div class="l">Generated Tasks</div></div>
        <div class="hstat"><div class="n">0%</div><div class="l">Init Answerability</div></div>
        <div class="hstat"><div class="n">100%</div><div class="l">Submit Coverage</div></div>
        <div class="hstat"><div class="n">{overall_score:.2f}</div><div class="l">Avg Expert Score</div></div>
      </div>
    </div>

    <!-- Introduction -->
    <div id="intro">
      <div class="section-h">Introduction</div>
      <div class="abstract">
        <strong>Spatial Training Room</strong> is an automated data factory that converts a 3D indoor
        reconstruction (Gaussian Splatting scene) + per-object AABB annotations + room polygon maps into
        <em>Active Perception Learning (APL)</em> training samples.
        Each sample is a 4D question:
        <code>(init_view, expert_action_sequence, target_view, question, answer)</code>.
        The initial viewpoint is deliberately selected so the question is <strong>unanswerable</strong>:
        the agent must actively navigate (pan, tilt, translate) to gather sufficient visual evidence
        before answering. The current release examples cover {len(sections_html)} task templates across occlusion,
        cross-room, counting, configuration, distance, and navigability.
        This page uses the <strong>waypoint-stitch</strong> expert trajectory batch:
        trajectory-memory tasks collect evidence across key views instead of requiring
        the final frame alone to contain every clue.
      </div>

    </div>

    <!-- Validation -->
    <div id="validation">
      <div class="section-h">Release Example Validation</div>
      <div class="val-sum">
        <div class="big">{total_tasks}</div>
        <p>
          Current examples: <code>{args.dataset_label}</code> generated {total_tasks} tasks from
          {len(sections_html)} release templates with average expert score {overall_score:.2f}
          and average trajectory length {overall_steps:.1f} steps.
        </p>
      </div>
    </div>

    <!-- Catalogue -->
    <div id="catalogue">
      <div class="section-h">Template Catalogue</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>Description</th><th>Category</th><th>Tasks</th><th>Avg Score</th><th>Avg Steps</th></tr></thead>
          <tbody>{cat_table_rows}</tbody>
        </table>
      </div>
      <p style="font-size:.75em;color:var(--text2);margin-top:8px">
        T14/T15/T16 not listed: scene lacks per-object <code>front_normal</code> / <code>label_side</code> annotations.
      </p>
    </div>

    <!-- Task Examples (per template) -->
    <div class="section-h" style="margin-top:48px">Task Examples - Release Templates</div>
    <p style="font-size:.85em;color:var(--text2);margin-bottom:24px">
      Each card shows the <strong>question + answer</strong>, then three key views:
      the init view the agent starts from (unanswerable), the top-down trajectory map,
      and the target view after navigation (answerable).
      Expand "Expert actions" to see the full action sequence.
    </p>

    {''.join(sections_html)}

  </div><!-- /main -->
</div><!-- /layout -->

</body>
</html>"""

    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[done] {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
