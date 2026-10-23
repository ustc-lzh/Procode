#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4: assemble padlock and readout probe sequences.

  4.1 Load the encode pool / readout and exclude used encodes
  4.2 Use select_encode to choose mutually dissimilar candidate encodes
  4.3 Use rank_select to choose targets per gene -> construct_padlock (switch encode if the secondary-structure check fails)
      -> construct_read_encode_whole_rounds
  4.4 merge_probe_all / filter_probe_merge
  4.5 Export color-filled encode and padlock tables

Implementation details:
  * check_second uses the standard-library implementation in target_region/oligoprop.py;
  * padlock arm lengths are derived from the actual target length,
    and Excel column offsets are computed accordingly instead of hard-coding 16+3/46+3/76+3/92+6;
  * random uses an independent Random(seed), so the same configuration is reproducible;
  * FASTA I/O uses the standard library, with no biopython dependency.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from sequence_utils import (Config, Resources, ensure_dirs, load_oligoprop, log,
                         longest_common_substring_length, make_reverse, progress,
                         read_fasta, select_widest_elements, step, sub, write_fasta)
from export_sequences import HAVE_OPENPYXL, write_colored_xlsx

__all__ = ["run_step4"]

PADLOCK_DIR = "01.padlock_probes"
ENCODE_DIR = "02.read_encode"
FILL_BLUE, FILL_ORANGE, FILL_GREEN, FILL_PURPLE = "33CCFF", "FF6633", "66FF33", "8D6891"


# --------------------------------------------------------------------------- #
# 4.2 encode selection
# --------------------------------------------------------------------------- #
def _hamming(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def _encode_ok(candidate: str, check_list: Sequence[str], ham_dis: int,
               edge_long_full: int, edge_ham_dis: int, edge_long: int,
               max_homopolymer: int, edge_len: int) -> bool:
    """Check one candidate encode against all sequence criteria."""
    if any(base * max_homopolymer in candidate for base in "ACGT"):
        return False
    if any(_hamming(candidate, other) < ham_dis for other in check_list):
        return False
    if any(longest_common_substring_length(candidate, other) > edge_long_full
           for other in check_list):
        return False
    edges = ([other[:edge_len] for other in check_list]
             + [other[-edge_len:] for other in check_list])
    head, tail = candidate[:edge_len], candidate[-edge_len:]
    if any(_hamming(head, other) < edge_ham_dis for other in edges):
        return False
    if any(_hamming(tail, other) < edge_ham_dis for other in edges):
        return False
    if any(longest_common_substring_length(head, other) > edge_long for other in edges):
        return False
    if any(longest_common_substring_length(tail, other) > edge_long for other in edges):
        return False
    return True


def _screen_worker(job):
    candidates, check_list, params = job
    return [_encode_ok(c, check_list, **params) for c in candidates]


def select_encode(encode_pool: Sequence[str], encode_used: Sequence[str],
                  select_number: int, rng: random.Random,
                  ham_dis: int = 18, edge_long_full: int = 8,
                  edge_ham_dis: int = 4, edge_long: int = 5,
                  max_homopolymer: int = 4, edge_len: int = 10,
                  workers: int = 1) -> List[str]:
    """Select mutually dissimilar encodes from an encoding pool.

    1. Exclude sequences containing max_homopolymer runs (AAAA/CCCC/GGGG/TTTT);
    2. Compared with selected/used sequences, require full-sequence Hamming distance >= ham_dis
       and longest common substring <= edge_long_full;
    3. Compared with the ends of existing sequences, require Hamming distance >= edge_ham_dis
       for each edge_len-nt end and longest common substring <= edge_long.

    This step compares the longest common substrings of tens of thousands of candidates and is
    a major Step4 bottleneck. In parallel mode, each batch is first screened concurrently against
    the **current** check_list, then accepted in the original order; only the few candidates
    accepted within the batch need an additional check. Results exactly match serial execution;
    only the most expensive comparisons are parallelized.
    """
    params = dict(ham_dis=ham_dis, edge_long_full=edge_long_full,
                  edge_ham_dis=edge_ham_dis, edge_long=edge_long,
                  max_homopolymer=max_homopolymer, edge_len=edge_len)
    selected: List[str] = []
    check_list: List[str] = [str(s) for s in encode_used]
    candidates = [str(c) for c in rng.sample(list(encode_pool), k=len(encode_pool))]

    if workers <= 1:
        for candidate in progress(candidates, desc="select encode", total=len(candidates)):
            if len(selected) >= select_number:
                break
            if _encode_ok(candidate, check_list, **params):
                selected.append(candidate)
                check_list.append(candidate)
    else:
        from concurrent.futures import ProcessPoolExecutor
        batch = max(workers * 40, 120)
        pos = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while pos < len(candidates) and len(selected) < select_number:
                chunk = candidates[pos:pos + batch]
                pos += len(chunk)
                size = (len(chunk) + workers - 1) // workers
                jobs = [(chunk[i:i + size], list(check_list), params)
                        for i in range(0, len(chunk), size)]
                flags: List[bool] = []
                for part in pool.map(_screen_worker, jobs):
                    flags.extend(part)
                # Accept in original order; newly accepted candidates in this batch need one more check.
                batch_new: List[str] = []
                for candidate, ok in zip(chunk, flags):
                    if len(selected) >= select_number:
                        break
                    if not ok:
                        continue
                    if batch_new and not _encode_ok(candidate, batch_new, **params):
                        continue
                    selected.append(candidate)
                    check_list.append(candidate)
                    batch_new.append(candidate)
                log("    selected encode %d/%d (scanned %d/%d)"
                    % (len(selected), select_number, pos, len(candidates)))

    if len(selected) < select_number:
        sub("Warning: selected only %d encode(s), target %d (consider relaxing step4.select_encode parameters)"
            % (len(selected), select_number))
    return selected


# --------------------------------------------------------------------------- #
# 4.3 assembly
# --------------------------------------------------------------------------- #
def rank_select(target_sub: pd.DataFrame, number: int = 4, isoform_name: str = "") -> List[str]:
    """Select number targets in tag_number order, distributing them across transcripts where possible."""
    if number >= target_sub.shape[0]:
        return list(target_sub["target"].values)

    if target_sub["tag_number"].max() == 1:
        picks = select_widest_elements(list(target_sub["loci_rank"].values), number)
    else:
        picks = []
        for tag_rank in sorted(target_sub["tag_number"].unique(), reverse=True):
            if len(isoform_name) == 0:
                scoped = target_sub
            elif tag_rank == 1:
                scoped = target_sub[target_sub["iso_name"].isin([isoform_name])]
                if scoped.shape[0] == 0:
                    continue
            else:
                scoped = target_sub
            block = scoped[scoped["tag_number"].isin([tag_rank])]
            if len(picks) + block.shape[0] >= number:
                picks += select_widest_elements(list(block["loci_rank"].values),
                                                number - len(picks))
                break
            picks += list(block["loci_rank"].values)
    return list(target_sub[target_sub["loci_rank"].isin(picks)]["target"].values)


class PadlockBuilder(object):
    def __init__(self, cfg: Config):
        self.oligo = load_oligoprop(cfg.target_region_dir)
        self.primer = str(cfg["step4"]["primer"])
        self.channel_names = list(cfg["step4"]["channel_names"])

    def check_second(self, seq: str) -> bool:
        """Return whether the full-length padlock has a hairpin or self-dimer (True = use another encode)."""
        hairpin, dimer = self.oligo.check_hairpin_dimer(str(seq).replace(" ", "").upper(),
                                                        8, 10)
        return bool(hairpin or dimer)

    def construct_padlock(self, gene: str, targets: Sequence[str], encode_seq: str,
                          output_path: str, check: bool = True) -> bool:
        """Return True when the sequence fails the secondary-structure check."""
        records: List[Tuple[str, str]] = []
        for index, target in enumerate(targets):
            half = len(target) // 2
            left = make_reverse(target[:half])
            right = make_reverse(target[half:])
            final_seq = "%s  %s  %s  %s" % (left, encode_seq, self.primer, right)
            if check and self.check_second(final_seq):
                return True
            records.append(("%s_n%dp" % (gene, index + 1), final_seq))
        out_dir = os.path.join(output_path, PADLOCK_DIR)
        ensure_dirs(out_dir)
        write_fasta(os.path.join(out_dir, "padlockseq.fasta"), records)
        return False

    def construct_read_encode(self, gene: str, encode_seq: str,
                              readout_seqs: Sequence[str], output_path: str) -> None:
        out_dir = os.path.join(output_path, ENCODE_DIR)
        ensure_dirs(out_dir)
        for index, readout in enumerate(readout_seqs):
            channel = self.channel_names[index] if index < len(self.channel_names) else str(index)
            record = ("%s_encode_%s" % (gene, channel),
                      "%s  %s" % (encode_seq, make_reverse(readout)))
            write_fasta(os.path.join(out_dir, "encode_read%d.fasta" % (index + 1)), [record])


# --------------------------------------------------------------------------- #
# 4.4 merge / deduplicate
# --------------------------------------------------------------------------- #
def merge_probe_all(path: str) -> None:
    out_dir = os.path.join(path, "00.merge")
    ensure_dirs(out_dir)
    for kind in (PADLOCK_DIR, ENCODE_DIR):
        merged: List[Tuple[str, str]] = []
        for entry in sorted(os.listdir(path)):
            if entry in ("00.merge", "01.merge_filter") or entry.endswith(".csv") \
                    or entry.endswith(".xlsx"):
                continue
            sub_dir = os.path.join(path, entry, kind)
            if not os.path.isdir(sub_dir):
                continue
            for fn in sorted(os.listdir(sub_dir)):
                merged.extend(read_fasta(os.path.join(sub_dir, fn), strip_space=False))
        write_fasta(os.path.join(out_dir, kind + "_all.fasta"), merged)


def filter_probe_merge(path: str) -> None:
    out_dir = os.path.join(path, "01.merge_filter")
    ensure_dirs(out_dir)
    for kind in (PADLOCK_DIR, ENCODE_DIR):
        records = read_fasta(os.path.join(path, "00.merge", kind + "_all.fasta"),
                             strip_space=False)
        seen: Set[str] = set()
        kept: List[Tuple[str, str]] = []
        for rec_id, seq in records:
            if seq in seen:
                continue
            seen.add(seq)
            kept.append((rec_id, seq))
        write_fasta(os.path.join(out_dir, kind + "_all.fasta"), kept)
        if len(kept) != len(records):
            sub("%s raw: %d; filter: %d" % (kind, len(records), len(kept)))
        else:
            sub("%s raw: %d" % (kind, len(records)))


# --------------------------------------------------------------------------- #
# 4.5 Excel export
# --------------------------------------------------------------------------- #
def transfer_fa_excel(fa_path: str, _type: str, save_path: str,
                      half_len: int = 19, encode_len: int = 30, primer_len: int = 30,
                      readout_len: int = 20) -> None:
    """Split the merged FASTA into columns, apply fills, and export an xlsx file.

    Note: spaces are removed when reading FASTA (matching Bio.SeqIO behavior), so padlock column
    offsets are contiguous [0,h) [h,h+30) [h+30,h+60) [h+60,2h+60); offsets are derived from
    actual lengths,
    Column offsets are derived from the actual target and readout lengths.
    """
    records = read_fasta(fa_path)          # strip_space=True
    fills: Dict[Tuple[int, int], str] = {}

    if _type == "encode":
        header = ["ID", "Sequence", "Encode", "Readout"]
        rows = [[rid, seq, seq[:encode_len], seq[-readout_len:]] for rid, seq in records]
        cycle = [FILL_BLUE, FILL_ORANGE, FILL_GREEN, FILL_PURPLE]
        for r in range(len(rows)):
            if r % 4 == 0:
                fills[(r, 2)] = FILL_BLUE          # Mark Encode once every 4 rows.
            fills[(r, 3)] = cycle[r % 4]           # Use one color for each of the 4 Readout channels.
    elif _type == "padlock":
        header = ["ID", "Sequence", "Target_left", "Encode", "Primer", "Target_right"]
        a, b, c = half_len, half_len + encode_len, half_len + encode_len + primer_len
        rows = [[rid, seq, seq[:a], seq[a:b], seq[b:c], seq[c:c + half_len]]
                for rid, seq in records]
        for r in range(len(rows)):
            fills[(r, 4)] = FILL_GREEN             # Color the entire Primer column green.
    else:
        raise ValueError("transfer_fa_excel: _type must be 'encode' / 'padlock'")

    write_colored_xlsx(save_path, header, rows, fills)
    pd.DataFrame(rows, columns=header).to_csv(
        os.path.splitext(save_path)[0] + ".csv", index=False, encoding="utf-8-sig")
    sub("%s -> %s (%d rows%s)"
        % (_type, save_path, len(rows), "" if HAVE_OPENPYXL else ", built-in xlsx writer"))


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def run_step4(cfg: Config, res: Resources, pool: Optional[pd.DataFrame] = None) -> None:
    step("Step4  Assemble probes")
    params = cfg["step4"]
    rng = random.Random(cfg.get("seed", 0))
    res.check_disk(cfg.work_dir, "Before starting Step4")

    # ---- Target pool ----------------------------------------------------- #
    if pool is None:
        pool_path = os.path.join(cfg.pool_dir, "%s.%s.target_pool.csv"
                                 % (cfg.sample_name, cfg.name))
        if not os.path.isfile(pool_path):
            raise FileNotFoundError("Target pool %s not found; run step3 first" % pool_path)
        pool = pd.read_csv(pool_path)
    genes = list(pd.unique(pool["gene_name"]))
    sub("Genes to assemble: %d; targets: %d" % (len(genes), pool.shape[0]))

    # ---- 4.1 encode / readout ------------------------------------------- #
    encode_path = os.path.join(cfg.data_web, "encode",
                               "BASSFISH_encoding_sequence_%s.txt" % cfg.species)
    readout_path = os.path.join(cfg.data_web, "readout", str(params["readout_file"]))
    for path in (encode_path, readout_path):
        if not os.path.isfile(path):
            raise FileNotFoundError("File not found: %s" % path)

    encode_pool = [seq for _rid, seq in read_fasta(encode_path)]
    rng.shuffle(encode_pool)
    readout_all = [seq for _rid, seq in read_fasta(readout_path)]
    # Use the first readouts for the requested channels; the number of channel_names is the channel count.
    n_channel = len(params.get("channel_names") or []) or len(readout_all)
    readout_seqs = readout_all[:n_channel]
    sub("encode pool %d / readout %d (%d channels used)"
        % (len(encode_pool), len(readout_all), len(readout_seqs)))

    # Exclude the full readout set: unselected readouts must not be chosen as encodes or collide later.
    encode_used: List[str] = list(readout_all) + list(params.get("encode_used_extra") or [])
    encode_used = list(dict.fromkeys(encode_used))
    encode_pool_del = [seq for seq in encode_pool if seq not in set(encode_used)]

    # ---- 4.2 Select encodes ---------------------------------------------- #
    need = len(genes) + int(params["encode_extra"])
    sub("Need %d encode(s) (%d genes + %d extra)" % (need, len(genes), params["encode_extra"]))
    se = params["select_encode"]
    encode_candidate = select_encode(
        encode_pool_del, list(params.get("encode_used_new") or []), need, rng,
        ham_dis=int(se["ham_dis"]), edge_long_full=int(se["edge_long_full"]),
        edge_ham_dis=int(se["edge_ham_dis"]), edge_long=int(se["edge_long"]),
        max_homopolymer=int(se["max_homopolymer"]), edge_len=int(se["edge_len"]),
        workers=res.current_threads())
    if not encode_candidate:
        raise RuntimeError("No encode selected; consider relaxing step4.select_encode parameters")
    sub("Selected %d encode(s)" % len(encode_candidate))

    # ---- 4.3 Assemble each gene ------------------------------------------ #
    builder = PadlockBuilder(cfg)
    ensure_dirs(cfg.output_dir)
    available = list(encode_candidate)
    assign_rows: List[Dict[str, object]] = []

    default_per_gene = int(params["probe_per_gene"])
    per_gene_map = {str(k): int(v) for k, v in (params.get("probe_per_gene_map") or {}).items()}
    if per_gene_map:
        sub("Expression-fitted counts: %d genes, %d ~ %d probes"
            % (len(per_gene_map), min(per_gene_map.values()), max(per_gene_map.values())))

    for gene in progress(genes, desc="assemble genes", total=len(genes)):
        want_per_gene = per_gene_map.get(gene, default_per_gene)
        target_sub = pool[pool["gene_name"].isin([gene])]
        targets = rank_select(target_sub, number=want_per_gene)
        if len(targets) == 0:
            sub("  %s has no usable target; skipping" % gene)
            continue
        out_path = os.path.join(cfg.output_dir, gene)

        chosen: Optional[str] = None
        forced = False
        for _id, candidate in enumerate(list(available)):
            if not builder.construct_padlock(gene, targets, candidate, out_path):
                chosen = candidate
                break
        if chosen is None:
            # If all candidates fail, retain the last one as a fallback.
            chosen = available[-1] if available else encode_candidate[-1]
            builder.construct_padlock(gene, targets, chosen, out_path, check=False)
            forced = True
            sub("  %s: all candidate encodes have secondary structures; forced to use the last one" % gene)
        if chosen in available:
            available.remove(chosen)

        builder.construct_read_encode(gene, chosen, readout_seqs, out_path)
        assign_rows.append({"gene_name": gene,
                            "n_padlock": len(targets),
                            "n_expected": want_per_gene,
                            "status": "ok" if len(targets) >= want_per_gene else "too_few",
                            "encode": chosen, "forced": forced})

    if not assign_rows:
        raise RuntimeError("Step4: no genes were assembled")
    thin = [r for r in assign_rows if r["status"] == "too_few"]
    if thin:
        sub("The following genes have fewer padlocks than expected (assembled using available counts): %s"
            % ", ".join("%s(%d/%d)" % (r["gene_name"], r["n_padlock"], r["n_expected"])
                        for r in thin))
    sub("Assembled %d genes; %d spare encodes remaining" % (len(assign_rows), len(available)))

    # ---- 4.4 Merge / deduplicate ----------------------------------------- #
    merge_probe_all(cfg.output_dir)
    filter_probe_merge(cfg.output_dir)

    # ---- 4.5 Export ------------------------------------------------------- #
    sample_target = str(pool["target"].iloc[0])
    half_len = len(sample_target) // 2
    encode_len = len(assign_rows[0]["encode"])          # type: ignore[arg-type]
    primer_len = len(str(params["primer"]))
    readout_len = len(readout_seqs[0]) if readout_seqs else 20

    transfer_fa_excel(os.path.join(cfg.output_dir, "01.merge_filter", ENCODE_DIR + "_all.fasta"),
                      "encode", os.path.join(cfg.output_dir, "readout_sequences.xlsx"),
                      half_len, encode_len, primer_len, readout_len)
    transfer_fa_excel(os.path.join(cfg.output_dir, "01.merge_filter", PADLOCK_DIR + "_all.fasta"),
                      "padlock", os.path.join(cfg.output_dir, "padlock_probes.xlsx"),
                      half_len, encode_len, primer_len, readout_len)

    ensure_dirs(cfg.report_dir)
    assign_path = os.path.join(cfg.report_dir, "step4_gene_encode.csv")
    pd.DataFrame(assign_rows).to_csv(assign_path, index=False, encoding="utf-8-sig")
    sub("Gene-encode mapping: %s" % assign_path)

    readout_rc_path = os.path.join(cfg.output_dir, "readout_reverse_complement.txt")
    with open(readout_rc_path, "w", encoding="utf-8", newline="\n") as fh:
        for index, readout in enumerate(readout_seqs):
            channel = (params["channel_names"][index]
                       if index < len(params["channel_names"]) else str(index))
            fh.write("%s\t%s\t%s\n" % (channel, readout, make_reverse(readout)))
    sub("Readout reverse complements: %s" % readout_rc_path)
    res.report(cfg.work_dir)
