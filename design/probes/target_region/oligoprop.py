#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 oligoprop.py — a pure-Python reimplementation of MATLAB Bioinformatics Toolbox ``oligoprop``.

Uses only the standard library (no numpy / biopython) and matches MATLAB R2023a
``oligoprop`` position by position. It is used by probe_design.py for GC / Tm /
hairpin / dimer checks.

Public interface
----------------
- ``gc_and_mean_tm(seq)``      -> (GC percentage, MATLAB ``mean(oligo.Tm)``)
- ``check_hairpin_dimer(seq, min_hpin_bases, dimer_length, min_hpin_loop=2)``
                               -> (whether Hairpins exist, whether Dimers exist)
- ``oligoprop(seq, ...)``      -> dict with MATLAB-compatible field names (for debugging/validation)

MATLAB defaults: HPBASE=4, HPLOOP=2, DIMERLENGTH=4, TEMP=25, SALT=0.05,
PRIMERCONC=50e-6; probe_filter1.m explicitly uses ``'HPBase', maxstem`` and
``'Dimerlength', maxdm``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

__all__ = ["gc_and_mean_tm", "check_hairpin_dimer", "oligoprop"]

# A=1, C=2, G=3, T=4, N=15 (matching MATLAB nt2int encoding).
_NT2NUM = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 15}

_DEFAULT_SALT = 0.05          # mol/L
_DEFAULT_PRIMER_CONC = 50e-6  # mol/L
_DEFAULT_TEMP = 25.0          # degrees Celsius

# --------------------------------------------------------------------------- #
# Nearest-neighbor thermodynamic parameter tables (Panjkovich & Melo,
# Bioinformatics 21(6):711-722, 2004).
# Rows and columns are A, C, G, T (the N version appends a fifth row/column).
# --------------------------------------------------------------------------- #
_BRES86_H = ((-9.1, -6.5, -7.8, -8.6), (-5.8, -11.0, -11.9, -7.8),
             (-5.6, -11.1, -11.0, -6.5), (-6.0, -5.6, -5.8, -9.1))
_BRES86_S = ((-24.0, -17.3, -20.8, -23.9), (-12.9, -26.6, -27.8, -20.8),
             (-13.5, -26.7, -26.6, -17.3), (-16.9, -13.5, -12.9, -24.0))
_SANT96_H = ((-8.4, -8.6, -6.1, -6.5), (-7.4, -6.7, -10.1, -6.1),
             (-7.7, -11.1, -6.7, -8.6), (-6.3, -7.7, -7.4, -8.4))
_SANT96_S = ((-23.6, -23.0, -16.1, -18.8), (-19.3, -15.6, -25.5, -16.1),
             (-20.3, -28.4, -15.6, -23.0), (-18.5, -20.3, -19.3, -23.6))
_SANT98_H = ((-7.9, -8.4, -7.8, -7.2), (-8.5, -8.0, -10.6, -7.8),
             (-8.2, -9.8, -8.0, -8.4), (-7.2, -8.2, -8.5, -7.9))
_SANT98_S = ((-22.2, -22.4, -21.0, -20.4), (-22.7, -19.9, -27.2, -21.0),
             (-22.2, -24.4, -19.9, -22.4), (-21.3, -22.2, -22.7, -22.2))
_SUGI96_H = ((-8.0, -9.4, -6.6, -5.6), (-8.2, -10.9, -11.8, -6.6),
             (-8.8, -10.5, -10.9, -9.4), (-6.6, -8.8, -8.2, -8.0))
_SUGI96_S = ((-21.9, -25.5, -16.4, -15.2), (-21.0, -28.4, -29.0, -16.4),
             (-23.5, -26.4, -28.4, -25.5), (-18.4, -23.5, -21.0, -21.9))

_BRES86_H_N = ((-9.1, -6.5, -7.8, -8.6, -8.0), (-5.8, -11.0, -11.9, -7.8, -9.125),
               (-5.6, -11.1, -11.0, -6.5, -8.55), (-6.0, -5.6, -5.8, -9.1, -6.625),
               (-6.625, -8.55, -9.125, -8.0, -8.075))
_BRES86_S_N = ((-24.0, -17.3, -20.8, -23.9, -21.5), (-12.9, -26.6, -27.8, -20.8, -22.025),
               (-13.5, -26.7, -26.6, -17.3, -21.025), (-16.9, -13.5, -12.9, -24.0, -16.825),
               (-16.825, -21.025, -22.025, -21.5, -20.3438))
_SANT96_H_N = ((-8.4, -8.6, -6.1, -6.5, -7.4), (-7.4, -6.7, -10.1, -6.1, -7.575),
               (-7.7, -11.1, -6.7, -8.6, -8.525), (-6.3, -7.7, -7.4, -8.4, -7.45),
               (-7.45, -8.525, -7.575, -7.4, -7.7375))
_SANT96_S_N = ((-23.6, -23.0, -16.1, -18.8, -20.375), (-19.3, -15.6, -25.5, -16.1, -19.125),
               (-20.3, -28.4, -15.6, -23.0, -21.825), (-18.5, -20.3, -19.3, -23.6, -20.425),
               (-20.425, -21.825, -19.125, -20.375, -20.4375))
_SANT98_H_N = ((-7.9, -8.4, -7.8, -7.2, -7.825), (-8.5, -8.0, -10.6, -7.8, -8.725),
               (-8.2, -9.8, -8.0, -8.4, -8.6), (-7.2, -8.2, -8.5, -7.9, -7.95),
               (-7.95, -8.6, -8.725, -7.825, -8.275))
_SANT98_S_N = ((-22.2, -22.4, -21.0, -20.4, -21.5), (-22.7, -19.9, -27.2, -21.0, -22.7),
               (-22.2, -24.4, -19.9, -22.4, -22.225), (-21.3, -22.2, -22.7, -22.2, -22.1),
               (-22.1, -22.225, -22.7, -21.5, -22.1313))
_SUGI96_H_N = ((-8.0, -9.4, -6.6, -5.6, -7.4), (-8.2, -10.9, -11.8, -6.6, -9.375),
               (-8.8, -10.5, -10.9, -9.4, -9.9), (-6.6, -8.8, -8.2, -8.0, -7.9),
               (-7.9, -9.9, -9.375, -7.4, -8.6437))
_SUGI96_S_N = ((-21.9, -25.5, -16.4, -15.2, -19.75), (-21.0, -28.4, -29.0, -16.4, -23.7),
               (-23.5, -26.4, -28.4, -25.5, -25.95), (-18.4, -23.5, -21.0, -21.9, -21.2),
               (-21.2, -25.95, -23.7, -19.75, -22.65))

_TABLES = ((_BRES86_H, _BRES86_S), (_SANT96_H, _SANT96_S),
           (_SANT98_H, _SANT98_S), (_SUGI96_H, _SUGI96_S))
_TABLES_N = ((_BRES86_H_N, _BRES86_S_N), (_SANT96_H_N, _SANT96_S_N),
             (_SANT98_H_N, _SANT98_S_N), (_SUGI96_H_N, _SUGI96_S_N))

# Collapse the eight 4x4 tables into a "dinucleotide -> 8 values" lookup;
# the hot loop performs one index operation.
_PAIR_LUT: List[Tuple[float, ...]] = []
for _a in range(4):
    for _b in range(4):
        _PAIR_LUT.append(tuple(
            v for (_H, _S) in _TABLES for v in (_H[_a][_b], _S[_a][_b])
        ))

_MOL_WEIGHT = (313.21, 289.18, 329.21, 304.2)  # A, C, G, T


def _to_nums(seq: str) -> List[int]:
    """Convert a sequence to numeric codes; raise ValueError for non-ACGTN characters
    (matching MATLAB IncorrectSequenceType).
    """
    try:
        return [_NT2NUM[c] for c in seq]
    except KeyError as exc:
        raise ValueError("IncorrectSequenceType: %r" % (exc.args[0],)) from None


def _is_self_complementary(nums: Sequence[int]) -> bool:
    n = len(nums)
    for i in range(n):
        if 5 - nums[i] != nums[n - 1 - i]:
            return False
    return True


def _near_neigh(nums: Sequence[int], self_comp: bool, n_flag: bool
                ) -> Tuple[List[List[float]], List[List[float]]]:
    """MATLAB oligoprop's internal nearneigh: return
    ``(NN[4][2], NNdelta[4][2])`` with columns ``[dH, dS]``.
    """
    if not n_flag:
        seq = nums
        nn = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        lut = _PAIR_LUT
        prev = seq[0] - 1
        for i in range(1, len(seq)):
            cur = seq[i] - 1
            v = lut[prev * 4 + cur]
            nn[0][0] += v[0]; nn[0][1] += v[1]
            nn[1][0] += v[2]; nn[1][1] += v[3]
            nn[2][0] += v[4]; nn[2][1] += v[5]
            nn[3][0] += v[6]; nn[3][1] += v[7]
            prev = cur
    else:
        # Replace 'N' code 15 with 5 and use the 5x5 averaged parameter tables.
        seq = [5 if v == 15 else v for v in nums]
        nn = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        for i in range(len(seq) - 1):
            a, b = seq[i] - 1, seq[i + 1] - 1
            for t in range(4):
                H, S = _TABLES_N[t]
                nn[t][0] += H[a][b]
                nn[t][1] += S[a][b]

    at_corr = ((0.0, -20.13), (0.0, -9.0), (0.0, 0.0), (0.6, -9.0))
    gc_corr = ((0.0, -16.77), (0.0, -5.9), (0.0, 0.0), (0.6, -9.0))
    sym_corr = ((0.0, -1.34), (0.0, -1.4), (0.0, -1.4), (0.0, -1.4))

    if not n_flag:
        corr = at_corr if all(v == 1 or v == 4 for v in seq) else gc_corr
        for t in range(4):
            nn[t][0] += corr[t][0]
            nn[t][1] += corr[t][1]
        if self_comp:
            for t in range(4):
                nn[t][0] += sym_corr[t][0]
                nn[t][1] += sym_corr[t][1]
        # 5' terminal initiation correction (Sant98 only, row 3).
        if seq[0] in (2, 3):
            nn[2][0] += 0.1; nn[2][1] += -2.8
        elif seq[0] in (1, 4):
            nn[2][0] += 2.3; nn[2][1] += 4.1
        # 3' terminal initiation correction.
        if seq[-1] in (2, 3):
            nn[2][0] += 0.1; nn[2][1] += -2.8
        elif seq[-1] in (1, 4):
            nn[2][0] += 2.3; nn[2][1] += 4.1
        nn_delta = [[0.0, 0.0] for _ in range(4)]
        return nn, nn_delta

    # With 'N': NN1 treats every N as G/C; NN2 treats every N as A/T.
    nn1 = [row[:] for row in nn]
    nn2 = [row[:] for row in nn]
    if all(v in (1, 4, 5) for v in seq):
        for t in range(4):
            nn1[t][0] += at_corr[t][0]
            nn1[t][1] += at_corr[t][1]
    else:
        for t in range(4):
            nn2[t][0] += gc_corr[t][0]
            nn2[t][1] += gc_corr[t][1]
    if self_comp:
        for t in range(4):
            nn1[t][0] += sym_corr[t][0]; nn1[t][1] += sym_corr[t][1]
            nn2[t][0] += sym_corr[t][0]; nn2[t][1] += sym_corr[t][1]
    if seq[0] in (2, 3, 5):
        nn2[2][0] += 0.1; nn2[2][1] += -2.8
    elif seq[0] in (1, 4, 5):
        nn1[2][0] += 2.3; nn1[2][1] += 4.1
    if seq[-1] in (2, 3, 5):
        nn2[2][0] += 0.1; nn2[2][1] += -2.8
    elif seq[-1] in (1, 4, 5):
        nn1[2][0] += 2.3; nn1[2][1] += 4.1

    nn_avg = [[(nn1[t][j] + nn2[t][j]) / 2.0 for j in range(2)] for t in range(4)]
    nn_delta = [[abs(nn1[t][j] - nn2[t][j]) / 2.0 for j in range(2)] for t in range(4)]
    return nn_avg, nn_delta


def _tm_vector(nums: Sequence[int], base_num: Sequence[int], salt: float,
               primer_conc: float, n_flag: bool
               ) -> Tuple[List[float], List[float], List[List[float]], List[List[float]]]:
    """Return ``(Tm[6], Tmdelta, NN, NNdelta)``; Tm = [basic, saltadj, NN x4]."""
    self_comp = _is_self_complementary(nums)
    b = 1.0 if self_comp else 4.0
    n_total = sum(base_num)
    log10_salt = math.log10(salt)
    c_a, c_c, c_g, c_t, c_n = base_num
    nn, nn_delta = _near_neigh(nums, self_comp, n_flag)
    rt = 1.9872 * math.log(primer_conc / b)

    if not n_flag:
        if n_total < 14:
            basic = 2.0 * (c_a + c_t) + 4.0 * (c_c + c_g)
            saltadj = basic - 16.6 * math.log10(0.05) + 16.6 * log10_salt
        else:
            basic = 64.9 + 41.0 * ((c_g + c_c - 16.4) / n_total)
            saltadj = (100.5 + 41.0 * ((c_g + c_c) / n_total)
                       - 820.0 / n_total + 16.6 * log10_salt)
        tms = [nn[t][0] * 1000.0 / (nn[t][1] + rt) + 16.6 * log10_salt - 273.15
               for t in range(4)]
        tm_delta = [0.0] * 6
    else:
        if n_total < 14:
            basic = 2.0 * (c_a + c_t) + 4.0 * (c_c + c_g) + 3.0 * c_n
            saltadj = basic - 16.6 * math.log10(0.05) + 16.6 * log10_salt
            tm_delta = [float(c_n), 3.0 * c_n]
        else:
            basic = 64.9 + 41.0 * (c_g + c_c - 16.4 + c_n / 2.0) / n_total
            saltadj = (100.5 - 820.0 / n_total + 16.6 * log10_salt
                       + 41.0 * ((c_c + c_g + c_n / 2.0) / n_total))
            tm_delta = [0.5 * 41.0 * (c_n / n_total), 41.0 * 0.5 * (c_n / n_total)]
        tms = []
        for t in range(4):
            hi = ((nn[t][0] + nn_delta[t][0]) * 1000.0
                  / ((nn[t][1] + nn_delta[t][1]) + rt) + 16.6 * log10_salt - 273.15)
            lo = ((nn[t][0] - nn_delta[t][0]) * 1000.0
                  / ((nn[t][1] - nn_delta[t][1]) + rt) + 16.6 * log10_salt - 273.15)
            tms.append((hi + lo) * 0.5)
            tm_delta.append((hi - lo) * 0.5)

    return [basic, saltadj] + tms, tm_delta, nn, nn_delta


def gc_and_mean_tm(seq: str, salt: float = _DEFAULT_SALT,
                   primer_conc: float = _DEFAULT_PRIMER_CONC) -> Tuple[float, float]:
    """Fast path computing only ``oligo.GC`` and ``mean(oligo.Tm)`` needed by probe_filter1.m.

    Hairpins / Dimers / MolWeight / Thermo are skipped; results match full oligoprop.
    """
    nums = _to_nums(seq)
    n = len(nums)
    if n == 0:
        raise ValueError("empty sequence")
    c_a = nums.count(1); c_c = nums.count(2)
    c_g = nums.count(3); c_t = nums.count(4); c_n = nums.count(15)
    n_flag = c_n > 0
    if not n_flag:
        gc = 100.0 * (c_c + c_g) / n
    else:
        gc = 100.0 * (c_c + c_g + c_n / 2.0) / n
    tm, _, _, _ = _tm_vector(nums, (c_a, c_c, c_g, c_t, c_n), salt, primer_conc, n_flag)
    return gc, sum(tm) / len(tm)


def _pairing_column(nums: Sequence[int], comp: Sequence[int], col: int, length: int
                    ) -> List[int]:
    """Return column ``col`` of MATLAB oligoprop's self-complementarity matrix W.

    W[r, col] = 1 means position r complements position (col-r) (or either end is N).
    All pairs in one column satisfy r + q = col, representing one reverse self-alignment frame.
    """
    w = [0] * length
    lo = col - (length - 1)
    if lo < 0:
        lo = 0
    hi = col if col < length else length - 1
    for r in range(lo, hi + 1):
        nq = nums[col - r]
        if nq == comp[r] or nq == 15 or nums[r] == 15:
            w[r] = 1
    return w


def check_hairpin_dimer(seq: str, min_hpin_bases: int, dimer_length: int,
                        min_hpin_loop: int = 2) -> Tuple[bool, bool]:
    """Return (whether a hairpin exists, whether a self-dimer exists), equivalent to MATLAB
    ``~isempty(oligo.Hairpins)`` / ``~isempty(oligo.Dimers)``.

    - ``min_hpin_bases``  = MATLAB 'HPBase'      (minimum complementary stem bases; probe_filter1 passes maxstem)
    - ``dimer_length``    = MATLAB 'DimerLength' (minimum complementary dimer bases; passes maxdm)
    - ``min_hpin_loop``   = MATLAB 'HPLoop'      (minimum loop bases; default 2)
    """
    nums = _to_nums(seq)
    length = len(nums)
    if not (1 < min_hpin_bases < length):
        raise ValueError("BadHairpinBaseParam")
    if not (1 < min_hpin_loop < length):
        raise ValueError("BadHairpinLoopParam")
    if not (1 < dimer_length < length):
        raise ValueError("BadDimerLengthParam")

    comp = [5 - v for v in nums]
    k = min_hpin_bases
    has_hairpin = False
    has_dimer = False

    for col in range(2 * length - 1):
        w = _pairing_column(nums, comp, col, length)

        if not has_dimer:
            run = 0
            for v in w:
                if v:
                    run += 1
                    if run >= dimer_length:
                        has_dimer = True
                        break
                else:
                    run = 0

        if not has_hairpin:
            # MATLAB's conv(W(:,c), ones(1,k), 'same') == k marks rows centered at this
            # position with k consecutive complementary bases; equivalently enumerate
            # every all-one window of length k starting at s.
            first_s = -1
            last_s = -1
            run = 0
            for idx in range(length):
                if w[idx]:
                    run += 1
                    if run >= k:
                        s = idx - k + 1
                        if first_s < 0:
                            first_s = s
                        last_s = s
                else:
                    run = 0
            # A hairpin requires the row-index difference between stems to be at least
            # the stem length plus the minimum loop length.
            if first_s >= 0 and (last_s - first_s) >= k + min_hpin_loop:
                has_hairpin = True

        if has_hairpin and has_dimer:
            break

    return has_hairpin, has_dimer


def oligoprop(seq: str, min_hpin_loop: int = 2, min_hpin_bases: int = 4,
              temp: float = _DEFAULT_TEMP, salt: float = _DEFAULT_SALT,
              primer_conc: float = _DEFAULT_PRIMER_CONC,
              dimer_length: int = 4) -> Dict[str, object]:
    """Full version (including Hairpins / Dimers strings), with MATLAB oligoprop field names.

    The main workflow uses only :func:`gc_and_mean_tm` and :func:`check_hairpin_dimer`;
    this function is primarily for value-by-value comparison with MATLAB / older numpy code.
    """
    seq = str(seq).upper()
    nums = _to_nums(seq)
    length = len(nums)
    c_a = nums.count(1); c_c = nums.count(2)
    c_g = nums.count(3); c_t = nums.count(4); c_n = nums.count(15)
    base_num = (c_a, c_c, c_g, c_t, c_n)
    n_flag = c_n > 0

    if not n_flag:
        gc = 100.0 * (c_c + c_g) / length
        gc_delta = 0.0
        weight = sum(base_num[i] * _MOL_WEIGHT[i] for i in range(4)) - 61.96
        weight_delta = 0.0
    else:
        gc = 100.0 * (c_c + c_g + c_n / 2.0) / length
        gc_delta = 100.0 * 0.5 * (c_n / length)
        weight = (sum(base_num[i] * _MOL_WEIGHT[i] for i in range(4)) - 61.96
                  + c_n * (_MOL_WEIGHT[2] + _MOL_WEIGHT[1]) / 2.0)
        weight_delta = c_n * (_MOL_WEIGHT[2] - _MOL_WEIGHT[1]) / 2.0

    tm, tm_delta, nn, nn_delta = _tm_vector(nums, base_num, salt, primer_conc, n_flag)
    thermo = [row + [row[0] - (temp + 273.15) * (row[1] / 1000.0)] for row in nn]
    if not n_flag:
        thermo_delta = [row + [0.0] for row in nn_delta]
    else:
        thermo_delta = [row + [row[0] - (temp + 273.15) * (row[1] / 1000.0)]
                        for row in nn_delta]

    comp = [5 - v for v in nums]
    hairpins: List[str] = []
    dimers: List[str] = []
    lower = seq.lower()
    for col in range(2 * length - 1):
        w = _pairing_column(nums, comp, col, length)
        rendered = "".join(seq[i] if w[i] else lower[i] for i in range(length))

        run = 0
        for v in w:
            if v:
                run += 1
                if run >= dimer_length:
                    dimers.append(rendered)
                    break
            else:
                run = 0

        first_s = last_s = -1
        run = 0
        for idx in range(length):
            if w[idx]:
                run += 1
                if run >= min_hpin_bases:
                    s = idx - min_hpin_bases + 1
                    if first_s < 0:
                        first_s = s
                    last_s = s
            else:
                run = 0
        if first_s >= 0 and (last_s - first_s) >= min_hpin_bases + min_hpin_loop:
            hairpins.append(rendered)

    return {
        "GC": gc, "GCdelta": gc_delta,
        "Hairpins": hairpins, "Dimers": dimers,
        "MolWeight": weight, "MolWeightdelta": weight_delta,
        "Tm": tm, "Tmdelta": tm_delta,
        "Thermo": thermo, "Thermodelta": thermo_delta,
    }
