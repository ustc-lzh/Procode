# Probe and codebook design

The two entry
points are `probes/design_probes.py` for target-region and padlock probes and
`codebooks/design_codebook.py` for binary readout codebooks.

The probe workflow reads its working inputs from `--work-dir` (or the current
directory): `ref/` contains the species reference files and `data/web/` contains
the expression tables used for transcript selection. The target-region implementation
is in `probes/target_region/`. BLAST databases and the transcript FASTA are external
inputs. A JSON file may be supplied with `--config`, while
module defaults and command-line overrides work without one.

The target-region defaults retain a 38 nt binding length, Tm 52--70 °C, GC
40--65%, maximum dimer/stem lengths 10/8, and single-/dual-arm homology limits
15/26 nt.  The configured `mid_length=6` contributes an 8 nt window in the
total-match criterion; it is not an exact 6 nt probe window.  A candidate is
removed only when the total match and the required two-arm or central-window
conditions are met, as implemented in `target_region/probe_design.py`.

Probe-count allocation uses each gene's maximum expression and whole-reference
P5/P95 anchors, inverse linear mapping, `ceil`, and clipping to the requested
count interval.  Codebook construction retains the Hamming, channel, and
round constraints and the seeded simulated-annealing moves and objectives.

Probe assembly reads a pre-generated nucleotide encoding pool. The initial
generation and transcriptome screening of that sequence pool are not included.
The binary codebook program generates its own constrained codeword candidates;
it does not use the nucleotide pool.

NumPy and pandas are required. BLAST+, openpyxl, tqdm and matplotlib are used
by the corresponding screening, export and plotting steps.
