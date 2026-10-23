# Codebook workflow

Run `python design_codebook.py --help`.  Required inputs are an expression
table (`--expression`, rows are genes and columns are cell types) and an output
directory; optional `--abundance` supplies one abundance column and `--genes`
selects rows.  The driver builds constrained binary code candidates, selects a
balanced subset, and optionally runs seeded simulated annealing. Inputs are
expression and cell-type abundance tables, not nucleotide barcode sequences.
