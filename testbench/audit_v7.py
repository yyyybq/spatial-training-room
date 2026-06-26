"""Audit fixed_v7 JSONL files and produce a markdown quality table."""
from __future__ import annotations
import json, re, pathlib, sys

JSONL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "out" / "fixed_v7"

TEMPLATES = ["T01","T04","T05","T06","T08","T11","T13","T17","T18","T19",
             "T20","T21","T23","T24","T26","T27","T29","T32","T33"]

ACTION_NAMES = {
    0: "forward", 1: "backward", 2: "turn_left", 3: "turn_right",
    4: "move_left", 5: "move_right",
}
STRAFE = {4, 5}  # move_left, move_right
FORWARD_BACK = {0, 1}
TURN = {2, 3}

def classify_action(a):
    if isinstance(a, str):
        a = a.lower()
        if "left" in a and "turn" in a: return "turn"
        if "right" in a and "turn" in a: return "turn"
        if "forward" in a: return "forward"
        if "backward" in a or "back" in a: return "forward"
        if "left" in a or "right" in a: return "strafe"
        return "other"
    if a in STRAFE: return "strafe"
    if a in FORWARD_BACK: return "forward"
    if a in TURN: return "turn"
    return "other"


def analyze_question(q: str):
    issues = []
    if re.search(r'\bthat\b', q, re.I):
        issues.append("that-pronoun")
    # "a " followed by a plural noun (ends in s, not -ss/-us/-is)
    for m in re.finditer(r'\ba\s+(\w+)', q, re.I):
        w = m.group(1)
        if w.lower() not in ("a","an","the") and w.lower().endswith("s") and not w.lower().endswith(("ss","us","is","ness","ness")):
            issues.append(f"a-plural({w})")
    if re.search(r'\bother\b', q, re.I):
        issues.append("generic-other")
    return issues


rows = []
for tid in TEMPLATES:
    fp = JSONL_ROOT / f"{tid}.jsonl"
    if not fp.exists():
        rows.append((tid, 0, "-", "-", "-", "-", "MISSING"))
        continue
    tasks = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(tasks)
    that_count = 0
    a_plural_count = 0
    generic_count = 0
    all_actions = []
    all_issues = []
    for t in tasks:
        q = t.get("question","")
        issues = analyze_question(q)
        if "that-pronoun" in issues: that_count += 1
        a_plural_count += sum(1 for i in issues if i.startswith("a-plural"))
        if "generic-other" in issues: generic_count += 1
        all_issues.extend(issues)
        seq = t.get("action_sequence", [])
        all_actions.extend([classify_action(a) for a in seq])

    total_moves = sum(1 for a in all_actions if a in ("forward","strafe"))
    forward_count = sum(1 for a in all_actions if a == "forward")
    strafe_count = sum(1 for a in all_actions if a == "strafe")
    fwd_ratio = forward_count / total_moves if total_moves else 0
    orth_ratio = strafe_count / total_moves if total_moves else 0
    total_steps = len(all_actions)
    issue_str = ", ".join(sorted(set(all_issues))) if all_issues else "none"
    rows.append((tid, n, f"{orth_ratio:.2f}", f"{fwd_ratio:.2f}",
                 f"{that_count}/{n}", f"{a_plural_count}/{n}",
                 f"{generic_count}/{n}", total_steps, issue_str))

# Print markdown table
header = ("Template","Tasks","orth_ratio","fwd_ratio","that_q","a_plural_q","generic_q","total_steps","issues")
widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
widths = [max(w, len(h)) for w,h in zip(widths, header)]

def fmt_row(row):
    return "| " + " | ".join(str(v).ljust(widths[i]) for i,v in enumerate(row)) + " |"

print(fmt_row(header))
print("| " + " | ".join("-"*w for w in widths) + " |")
for r in rows:
    print(fmt_row(r))

# Summary
total_that = sum(int(r[4].split("/")[0]) for r in rows if "/" in str(r[4]))
total_a_plural = sum(int(r[5].split("/")[0]) for r in rows if "/" in str(r[5]))
total_generic = sum(int(r[6].split("/")[0]) for r in rows if "/" in str(r[6]))
all_orth = [float(r[2]) for r in rows if r[2] != "-"]
all_fwd = [float(r[3]) for r in rows if r[3] != "-"]
print(f"\n**Summary**: that_q={total_that}, a_plural_q={total_a_plural}, generic_q={total_generic}")
if all_orth:
    print(f"**Avg orth_ratio**={sum(all_orth)/len(all_orth):.3f}  "
          f"**Avg fwd_ratio**={sum(all_fwd)/len(all_fwd):.3f}")
