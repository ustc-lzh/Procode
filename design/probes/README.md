# Probe workflow

Run `python design_probes.py --help` for the stage and override options.  The
workflow reads `ref/`, `data/web/` and the reference databases from the selected
working directory. The target-region implementation is bundled in `target_region/`.
The optional JSON configuration
uses the same keys as `sequence_utils.DEFAULT_CONFIG`; no configuration file is
required for defaults and command-line overrides.

`probe_allocation.py` maps per-gene maximum expression to probe counts using
P5/P95 whole-reference anchors, inverse linear interpolation, ceiling, and
clipping.  The target-region implementation keeps the source defaults and
the total-match/dual-arm homology filters described in the parent README.
