# Release checklist

This checklist is scoped to a v0.1 internal/research release: installable
package, stable CLI entry point, reproducible core tests, and documented data
quality caveats.

## Required before tagging

- [ ] Confirm the project license and add a `LICENSE` file.
- [ ] Verify `README.md` renders correctly as UTF-8 in GitHub or the target
      release platform.
- [ ] Build and install the package in a clean environment:
      `python -m pip install -e .`
- [ ] Run the core regression checks:
      `python testbench/test_regression_core.py`
- [ ] Run the template schema/handler checks:
      `python testbench/test_items_6_and_7.py`
- [ ] Verify the console script starts:
      `spatial-room-factory --help`
- [ ] Run at least one smoke generation against a real or sample scene:
      `python testbench/smoke_test_template.py --scene <SCENE_DIR> --template T01 --n 1`
- [ ] Run a template sweep on the release sample scene and store the summary:
      `python testbench/sweep_all_templates.py --scene <SCENE_DIR> --n 1 --out out/release_smoke --timeout-per-template 120`
      By default this skips templates excluded from the current release mix
      (`T13`, `T19`, `T23`, `T24`, `T27`, `T32`). Use
      `--include-excluded` only for audits.
- [ ] Audit generated JSONL files:
      `python testbench/audit_trajectory_reliability.py out/release_smoke`
- [ ] Remove or exclude local generated artifacts from the source release
      (`out/`, rendered HTML galleries, debug logs, temporary probes).
- [ ] Confirm generated artifacts are not tracked by Git:
      `git ls-files out "*.html" "*.log" "*.egg-info" "__pycache__"`
      should print nothing for release-only generated files.
- [ ] Review the push diff:
      `git status --short` and `git diff --stat`.

## Known v0.1 caveats

- The source directory is named `spatial-training-room`, while the importable
  package is exposed as `spatial_training_room` through `pyproject.toml`.
- Rendering and model inference are optional stacks. Install `.[render]` or
  `.[inference]` only when those workflows are needed.
- Some visualization scripts still reference project-specific external modules
  such as `view_suite` and `ply_gaussian_loader`; these are not part of the
  core generation path.
- Trajectory-mode templates can require evidence collected across the path;
  their final frame alone may not be answerable by design.
- Current local audit found a T29 beam-sensitivity warning in one generated
  sample. Fix or document this before a public release.

## Suggested tag flow

1. Create a clean environment and install the package.
2. Run the required checks above.
3. Build source and wheel distributions:
   `python -m build`
4. Inspect the wheel contents for templates, configs, and docs.
5. Tag as `v0.1.0` only after the license and quality notes are complete.
