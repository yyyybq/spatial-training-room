"""
task_generation/apl_tasks/template_handler.py

TemplateHandler — single per-``template_id`` handle that bundles the things
that always travel together: the loaded ``TemplateSpec`` and its
``instantiator`` function.

Why this exists
===============
Before this module, every caller had to do two lookups:

    spec       = load_template("T05")            # YAML  -> TemplateSpec
    instantiate = INSTANTIATORS["T05"]            # Python -> instantiator fn

That's two parallel registries that you must keep in sync.  This module
exposes a single accessor — ``get_template_handler("T05")`` — that returns a
``TemplateHandler`` carrying both.

What this does NOT merge
========================
The other three registries are intentionally left alone because they are
**shared lookup tables keyed by name strings, not by template_id**:

    PREDICATE_REGISTRY  — keyed by predicate name (e.g. "Visible")
    CHOICE_REGISTRY     — keyed by choice generator name (e.g. "pair_labels")
    INIT_VALIDATORS     — keyed by YAML trigger field name

Many templates share the same predicate / choice generator / trigger field,
so folding them into a per-template dataclass would create duplication.
``TemplateHandler`` therefore wraps only the two registries that ARE
per-template (the YAML spec and the instantiator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ...evaluation import TemplateSpec, load_template


# Re-exported for convenience.  We import lazily inside the helper to avoid
# a circular import (template_active_generator imports this module's siblings).

@dataclass(frozen=True)
class TemplateHandler:
    """Everything per-template, in one place.

    Attributes
    ----------
    template_id : str
        The canonical id (e.g. ``"T05"``).
    spec : TemplateSpec
        Parsed YAML for this template (slots, trigger, question templates, …).
    instantiator : Callable
        ``fn(spec, scene_ctx, rng) -> Optional[task_instance]`` — the
        Python-side function that binds template variables to concrete IDs.
        ``None`` when the template has no registered instantiator yet.
    """

    template_id: str
    spec: TemplateSpec
    instantiator: Optional[Callable[..., Any]]


def get_template_handler(template_id: str) -> TemplateHandler:
    """Load the YAML spec and look up the instantiator in one call.

    Raises
    ------
    FileNotFoundError
        If no YAML file exists for *template_id*.
    """
    # Lazy import to avoid an import cycle:
    # template_active_generator imports from sibling modules at import time,
    # and those siblings have nothing to do with this convenience wrapper.
    from .template_active_generator import INSTANTIATORS

    spec = load_template(template_id)
    inst = INSTANTIATORS.get(template_id)
    return TemplateHandler(template_id=template_id, spec=spec, instantiator=inst)


def all_template_handlers() -> Dict[str, TemplateHandler]:
    """Return one ``TemplateHandler`` per registered instantiator.

    Useful for sweeps that want to iterate the implemented templates in one
    pass without juggling two registries.
    """
    from .template_active_generator import INSTANTIATORS

    out: Dict[str, TemplateHandler] = {}
    for tid in sorted(INSTANTIATORS):
        try:
            out[tid] = get_template_handler(tid)
        except FileNotFoundError:
            # An instantiator without a YAML is a config bug, but we don't
            # want a single missing file to break the whole iteration.
            continue
    return out


# ---------------------------------------------------------------------------
# Optional decorator — alias of ``register_instantiator`` whose name reads
# more naturally for new templates ("I am registering a TEMPLATE, which
# happens to also register its instantiator").
# ---------------------------------------------------------------------------

def register_template(template_id: str):
    """Decorator alias of ``register_instantiator(template_id)``.

    Prefer this name in new code; old call sites continue to work via the
    original decorator.
    """
    from .template_active_generator import register_instantiator
    return register_instantiator(template_id)
