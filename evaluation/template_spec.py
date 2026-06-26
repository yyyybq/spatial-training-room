"""
evaluation/template_spec.py
TemplateSpec and EvidenceSlot dataclasses + YAML loader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Locate the templates directory relative to this file
_TEMPLATES_ROOT = Path(__file__).parent.parent / "task_generation" / "templates"


@dataclass
class PredicateSpec:
    """One predicate entry inside an evidence slot."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PredicateSpec":
        return cls(name=d["name"], args=d.get("args", {}))


@dataclass
class EvidenceSlot:
    """
    One slot in a template.

    slot_id          : unique identifier within the template
    region_generator : name of the region sampler function (in region_generators.py)
    region_args      : kwargs forwarded to the region generator
    predicates       : list of PredicateSpec — ALL must pass (threshold=1.0) or
                       fraction ≥ threshold
    threshold        : fraction of predicates that must pass (default 1.0 = all)
    tier2_override   : optional name of a Tier-2 callable in QUALITY_REGISTRY
    """
    slot_id: str
    region_generator: str
    region_args: dict[str, Any] = field(default_factory=dict)
    predicates: list[PredicateSpec] = field(default_factory=list)
    threshold: float = 1.0
    tier2_override: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceSlot":
        return cls(
            slot_id=d["slot_id"],
            region_generator=d["region_generator"],
            region_args=d.get("region_args", {}),
            predicates=[PredicateSpec.from_dict(p) for p in d.get("predicates", [])],
            threshold=float(d.get("threshold", 1.0)),
            tier2_override=d.get("tier2_override"),
        )


@dataclass
class ActionConfig:
    move_m: float = 0.5
    turn_deg: float = 45.0
    look_deg: float = 15.0

    @classmethod
    def from_dict(cls, d: dict) -> "ActionConfig":
        return cls(
            move_m=float(d.get("move_m", 0.5)),
            turn_deg=float(d.get("turn_deg", 45.0)),
            look_deg=float(d.get("look_deg", 15.0)),
        )


@dataclass
class TemplateSpec:
    """
    Full specification of one APL active task template, loaded from YAML.
    """
    template_id: str
    name: str
    subclass: str
    priority: str                       # "P0" | "P1"
    description: str
    question_templates: list[str]
    answer_type: str                    # categorical | binary | ordinal | count
    answer_choices_generator: str
    gt_from: str
    evidence_slots: list[EvidenceSlot]
    coverage_aggregator: str = "all"   # "all" | "mean"
    coverage_mode: str = "submit"      # "submit" | "trajectory"
    min_coverage_for_credit: float = 1.0
    action_config: ActionConfig = field(default_factory=ActionConfig)
    max_steps: int = 10
    gamma: float = 0.95
    scene_requirements: dict[str, Any] = field(default_factory=dict)
    trigger: dict[str, Any] = field(default_factory=dict)
    init_view: dict[str, Any] = field(default_factory=dict)
    # P1 extra
    p1_requirement: Optional[str] = None
    # Pattern-based multi-slot (T06, T26, T32) — raw YAML preserved for generator
    evidence_slot_pattern: Optional[dict] = None
    # Honesty / mixing knobs (v3 §9 honest-downgrade)
    caveat: Optional[str] = None
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateSpec":
        # Apply unified ``checks:`` projection (Item 6) BEFORE reading legacy
        # fields, so that authors may write either style and existing YAML
        # files keep working untouched.
        d = _project_checks(d)
        slots_raw = d.get("evidence_slots", [])
        slots = [EvidenceSlot.from_dict(s) for s in slots_raw] if slots_raw else []
        coverage_mode = str(d.get("coverage_mode", "submit")).strip().lower()
        if coverage_mode not in ("submit", "trajectory"):
            raise ValueError(
                f"template {d.get('template_id', '?')}: coverage_mode must be "
                f"'submit' or 'trajectory', got {coverage_mode!r}"
            )
        return cls(
            template_id=d["template_id"],
            name=d["name"],
            subclass=d["subclass"],
            priority=d.get("priority", "P0"),
            description=d.get("description", ""),
            question_templates=d.get("question_templates", []),
            answer_type=d.get("answer_type", "categorical"),
            answer_choices_generator=d.get("answer_choices_generator", ""),
            gt_from=d.get("gt_from", ""),
            evidence_slots=slots,
            coverage_aggregator=d.get("coverage_aggregator", "all"),
            coverage_mode=coverage_mode,
            min_coverage_for_credit=float(d.get("min_coverage_for_credit", 1.0)),
            action_config=ActionConfig.from_dict(d.get("action_config", {})),
            max_steps=int(d.get("max_steps", 10)),
            gamma=float(d.get("gamma", 0.95)),
            scene_requirements=d.get("scene_requirements", {}),
            trigger=d.get("trigger", {}),
            init_view=d.get("init_view", {}) or {},
            p1_requirement=d.get("p1_requirement"),
            evidence_slot_pattern=d.get("evidence_slot_pattern"),
            caveat=d.get("caveat"),
            weight=float(d.get("weight", 1.0)),
        )


# ---------------------------------------------------------------------------
# Unified ``checks:`` schema (Item 6)
# ---------------------------------------------------------------------------
#
# New YAMLs MAY declare a single ``checks:`` list whose entries are tagged by
# ``when:`` (scene | init | target).  This is a *cosmetic* surface — at load
# time we project each entry into the legacy fields below, so generator code
# and the existing 19 YAML files keep working untouched.
#
#   when: scene  -> merged into ``scene_requirements`` (flat dict)
#   when: init   -> merged into ``trigger``            (flat dict)
#   when: target -> appended as a predicate to evidence_slots[slot=<slot>]
#                   (slot defaults to the first existing slot)
#
# Example (equivalent to T05's legacy form):
#
#   checks:
#     - {when: scene, name: min_objects, args: 3}
#     - {when: init,  name: max_both_pairs_visible_at_init, args: false}
#     - {when: target, slot: AB_view,
#        name: PairVisible, args: {obj_a: "{{a}}", obj_b: "{{b}}"}}
#
# Mixing ``checks:`` with the legacy fields is allowed; the projection MERGES
# (legacy entries win on key conflict so an explicit override is honoured).
# ---------------------------------------------------------------------------


def _project_checks(d: dict) -> dict:
    """Return a shallow copy of *d* with ``checks:`` projected into the legacy
    ``scene_requirements`` / ``trigger`` / ``evidence_slots`` fields.

    No-op when ``checks:`` is absent.
    """
    checks = d.get("checks")
    if not checks:
        return d
    if not isinstance(checks, list):
        raise ValueError(
            f"template {d.get('template_id', '?')}: 'checks' must be a list, "
            f"got {type(checks).__name__}"
        )

    out = dict(d)  # shallow copy; we only mutate top-level fields we add
    scene_req: dict = dict(out.get("scene_requirements", {}) or {})
    trigger:  dict = dict(out.get("trigger", {}) or {})
    # We keep evidence_slots in their list form and edit by slot_id.
    slots_raw: list = list(out.get("evidence_slots", []) or [])

    def _find_slot(slot_id: Optional[str]) -> dict:
        if not slots_raw:
            raise ValueError(
                f"template {out.get('template_id', '?')}: 'checks' entry has "
                f"when=target but template defines no evidence_slots"
            )
        if slot_id is None:
            return slots_raw[0]
        for s in slots_raw:
            if s.get("slot_id") == slot_id:
                return s
        raise ValueError(
            f"template {out.get('template_id', '?')}: 'checks' references "
            f"unknown evidence slot_id={slot_id!r}"
        )

    for i, entry in enumerate(checks):
        if not isinstance(entry, dict):
            raise ValueError(
                f"template {out.get('template_id', '?')}: checks[{i}] must "
                f"be a mapping, got {type(entry).__name__}"
            )
        when = entry.get("when")
        name = entry.get("name")
        args = entry.get("args", {})
        if when not in ("scene", "init", "target"):
            raise ValueError(
                f"template {out.get('template_id', '?')}: checks[{i}].when "
                f"must be scene|init|target, got {when!r}"
            )
        if not name:
            raise ValueError(
                f"template {out.get('template_id', '?')}: checks[{i}] missing 'name'"
            )

        if when == "scene":
            # Legacy ``scene_requirements`` is a flat dict {field: value}.
            # The author writes ``{name: min_objects, args: 3}``; we store
            # min_objects=3.  args may be a scalar or a dict — store as-is.
            scene_req.setdefault(name, args)
        elif when == "init":
            # Same flat-dict shape for ``trigger:``.
            trigger.setdefault(name, args)
        else:  # target
            slot = _find_slot(entry.get("slot"))
            preds = list(slot.get("predicates", []) or [])
            preds.append({"name": name, "args": args})
            slot["predicates"] = preds

    out["scene_requirements"] = scene_req
    out["trigger"] = trigger
    out["evidence_slots"] = slots_raw
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_yaml(template_id: str) -> Optional[Path]:
    """Search p0/ and p1/ directories for a YAML file starting with template_id."""
    for priority in ("p0", "p1"):
        folder = _TEMPLATES_ROOT / priority
        if folder.exists():
            for f in folder.glob(f"{template_id}_*.yaml"):
                return f
    return None


# ---------------------------------------------------------------------------
# Pattern slot expansion (shared by generator, potential, and coverage)
# ---------------------------------------------------------------------------

def _sub_i(value: Any, i: int) -> Any:
    """Recursively replace the literal token ``{i}`` with ``str(i)``."""
    if isinstance(value, str):
        return value.replace("{i}", str(i))
    if isinstance(value, dict):
        return {k: _sub_i(v, i) for k, v in value.items()}
    if isinstance(value, list):
        return [_sub_i(v, i) for v in value]
    return value


def expand_evidence_slots(
    spec: "TemplateSpec", task_instance: dict
) -> list[EvidenceSlot]:
    """
    Return the effective evidence slots for *spec* given *task_instance*.

    • If ``spec.evidence_slots`` is non-empty, return it unchanged.
    • Otherwise expand ``spec.evidence_slot_pattern`` by inferring how many
      slots to create from keys like ``prefix_0_id``, ``prefix_1_id`` … in
      *task_instance*, then substituting the ``{i}`` token throughout.
    """
    if spec.evidence_slots:
        return list(spec.evidence_slots)
    if not spec.evidence_slot_pattern:
        return []
    pattern = spec.evidence_slot_pattern
    slot_id_tmpl = str(pattern.get("slot_id", "slot_{i}"))
    prefix = slot_id_tmpl.replace("{i}", "")
    n = int(task_instance.get("num_evidence_slots", 0) or 0)
    if n <= 0:
        n = 0
        while f"{prefix}{n}_id" in task_instance:
            n += 1
        if n == 0:
            n = 1
    slots: list[EvidenceSlot] = []
    for i in range(n):
        slot_id = slot_id_tmpl.replace("{i}", str(i))
        region_args = _sub_i(pattern.get("region_args", {}) or {}, i)
        preds_raw = _sub_i(pattern.get("predicates", []) or [], i)
        preds = [PredicateSpec(name=p["name"], args=p.get("args", {}))
                 for p in preds_raw]
        slots.append(EvidenceSlot(
            slot_id=slot_id,
            region_generator=pattern.get("region_generator", ""),
            region_args=region_args,
            predicates=preds,
            threshold=float(pattern.get("threshold", 1.0)),
            tier2_override=pattern.get("tier2_override"),
        ))
    return slots


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_template(template_id: str) -> TemplateSpec:
    """Load a TemplateSpec from its YAML file by template_id (e.g., 'T01')."""
    path = _find_yaml(template_id)
    if path is None:
        raise FileNotFoundError(
            f"No YAML template found for {template_id!r} in {_TEMPLATES_ROOT}"
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return TemplateSpec.from_dict(data)


def load_all_templates(priority: Optional[str] = None) -> dict[str, TemplateSpec]:
    """
    Load all templates.

    Parameters
    ----------
    priority : 'P0', 'p0', 'P1', 'p1', or None (load all)
    """
    results: dict[str, TemplateSpec] = {}
    folders = []
    if priority is None:
        folders = ["p0", "p1"]
    else:
        folders = [priority.lower()]

    for folder_name in folders:
        folder = _TEMPLATES_ROOT / folder_name
        if not folder.exists():
            continue
        for yaml_file in sorted(folder.glob("*.yaml")):
            with open(yaml_file, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            spec = TemplateSpec.from_dict(data)
            results[spec.template_id] = spec

    return results
