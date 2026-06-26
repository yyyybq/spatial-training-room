#!/usr/bin/env python3
"""Build a single-page HTML review sheet for manually inspecting rendered tasks.

Usage:
    python testbench/build_review_html.py \
        --rendered-root out/fixed_v2_rendered \
        --jsonl-root   out/fixed_v2 \
        --out          out/fixed_v2_review.html
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path


def _img_b64(path: Path) -> str:
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def build(rendered_root: Path, jsonl_root: Path, out_html: Path) -> None:
    # Collect all template-level rendered folders
    sections: list[str] = []

    for tmpl_dir in sorted(rendered_root.iterdir()):
        if not tmpl_dir.is_dir():
            continue
        template_id = tmpl_dir.name          # e.g. "T04"

        # manifest is directly in tmpl_dir (flat structure)
        manifest_path = tmpl_dir / "manifest.json"
        # Also support old double-nested layout (tmpl_dir/template_id/)
        inner = tmpl_dir / template_id
        if not manifest_path.exists() and (inner / "manifest.json").exists():
            manifest_path = inner / "manifest.json"
            actual_root = inner
        else:
            actual_root = tmpl_dir

        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Load corresponding JSONL for questions / gt / choices
        jsonl_path = jsonl_root / f"{template_id}.jsonl"
        task_rows = _read_jsonl(jsonl_path)

        cards_html = []
        for t in manifest.get("tasks", []):
            idx = t["task_index"]
            task_row = task_rows[idx] if idx < len(task_rows) else {}

            question    = task_row.get("question", "(no question)")
            gt          = task_row.get("gt_answer") or task_row.get("answer", "?")
            choices     = task_row.get("choices") or []
            coverage    = task_row.get("coverage", 0.0)
            score       = task_row.get("score", 0.0)
            steps       = len(task_row.get("action_sequence") or [])
            tid_label   = task_row.get("template_id", template_id)
            actions     = t.get("action_descriptions") or []

            # Init, target, and top-down images
            init_png  = actual_root / t["task_dir"] / "init.png"
            tgt_png   = actual_root / t["task_dir"] / "target.png"
            topdn_png = actual_root / t["task_dir"] / "topdown.png"
            init_src  = _img_b64(init_png)  if init_png.exists()  else ""
            tgt_src   = _img_b64(tgt_png)   if tgt_png.exists()   else ""
            topdn_src = _img_b64(topdn_png) if topdn_png.exists() else ""

            # All step frames
            frames_html = ""
            frames_dir = actual_root / t["task_dir"]
            for fname in t.get("frames", []):
                fp = frames_dir / fname
                if fp.exists():
                    src = _img_b64(fp)
                    frames_html += f'<img src="{src}" style="width:128px;height:128px;margin:2px;border:1px solid #999">'

            choices_str = " | ".join(f"<b>{c}</b>" if c == gt else c for c in choices)
            actions_str = "<br>".join(actions) if actions else "(no action descriptions)"

            card = f"""
<div style="border:2px solid #ccc;border-radius:8px;padding:12px;margin:10px 0;background:#fafafa">
  <div style="font-size:13px;color:#555;margin-bottom:6px">
    <b>{tid_label}</b> &nbsp; task #{idx} &nbsp;|&nbsp; cov={coverage:.2f} &nbsp; score={score:.2f} &nbsp; steps={steps}
  </div>
  <div style="font-size:15px;font-weight:bold;margin-bottom:8px">{question}</div>
  <div style="margin-bottom:8px">Choices: {choices_str}</div>
  <div style="margin-bottom:10px;color:#090">GT answer: <b>{gt}</b></div>
  <table><tr>
    <td style="text-align:center;padding-right:16px">
      {'<img src="' + init_src + '" style="width:256px;height:256px;border:2px solid #e55">' if init_src else '(no init)'}
      <div style="font-size:11px;color:#e55">INIT (unanswerable)</div>
    </td>
    <td style="text-align:center;padding-right:16px">
      {'<img src="' + tgt_src + '" style="width:256px;height:256px;border:2px solid #0a5">' if tgt_src else '(no target)'}
      <div style="font-size:11px;color:#0a5">TARGET (answerable)</div>
    </td>
    <td style="text-align:center">
      {'<img src="' + topdn_src + '" style="width:256px;height:256px;border:2px solid #55a">' if topdn_src else '(no topdown)'}
      <div style="font-size:11px;color:#55a">FLOOR PLAN (&#x25CF;red=INIT &#x25CF;green=END &#x25CF;blue=steps)</div>
    </td>
  </tr></table>
  <details style="margin-top:8px">
    <summary style="cursor:pointer;color:#36c">All trajectory frames ({len(t.get('frames', []))})</summary>
    <div style="margin-top:6px">{frames_html}</div>
  </details>
  <details style="margin-top:6px">
    <summary style="cursor:pointer;color:#36c">Action sequence</summary>
    <div style="font-size:12px;color:#444;margin-top:4px">{actions_str}</div>
  </details>
</div>"""
            cards_html.append(card)

        if not cards_html:
            continue

        section = f"""
<h2 style="background:#2c3e50;color:#fff;padding:8px 14px;border-radius:6px;margin-top:32px">{template_id}</h2>
{"".join(cards_html)}"""
        sections.append(section)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>APL fixed_v2 Review</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }}
  summary {{ font-size: 13px; }}
</style>
</head>
<body>
<h1>APL fixed_v2 — Manual Review</h1>
<p style="color:#555">Scene: 0267_840790 &nbsp;|&nbsp; seed=42 &nbsp;|&nbsp; 3 tasks per template<br>
CPU-splat rendering (no CUDA). Init view = red border; target view = green border.</p>
{"".join(sections)}
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[done] {out_html}  ({len(sections)} templates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered-root", default="out/fixed_v2_rendered")
    ap.add_argument("--jsonl-root",    default="out/fixed_v2")
    ap.add_argument("--out",           default="out/fixed_v2_review.html")
    args = ap.parse_args()
    build(
        rendered_root=Path(args.rendered_root),
        jsonl_root=Path(args.jsonl_root),
        out_html=Path(args.out),
    )
