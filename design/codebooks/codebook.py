#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct constrained codebooks and optimize their allocation.

The public functions generate binary code candidates, select balanced subsets,
and run the seeded simulated-annealing objective used by the command-line
driver.  Plotting and progress bars remain optional dependencies.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:                       # Plotting is optional.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:          # noqa: BLE001
    plt = None
    HAVE_MPL = False


def _print(*args, **kwargs):
    """Send progress text through the module logger."""
    logger.info(" ".join(str(a) for a in args))


class _NoTqdm(object):
    def __init__(self, iterable=None, *a, **k):
        self.iterable = iterable if iterable is not None else []
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def update(self, *a, **k):
        pass
    def close(self):
        pass
    def set_description(self, *a, **k):
        pass


try:
    from tqdm import tqdm as _tqdm
except Exception:          # noqa: BLE001
    _tqdm = _NoTqdm

tqdm = _tqdm

# --------------------------------------------------------------------------- #
# Codebook construction and validation.
# --------------------------------------------------------------------------- #
def get_new_col_codebook(_small_round, rounds_per=4, add_sum=0):
    max_col_index = []
    for i in range(100):
        i += add_sum
        if rounds_per == 4:
            max_col_index = max_col_index + ['488_'+str(i+1), '561_'+str(i+1),
                                         '647_'+str(i+1), '750_'+str(i+1)]
        else:
            ## Exclude 488.
            max_col_index = max_col_index + ['561_'+str(i+1),
                                         '647_'+str(i+1), '750_'+str(i+1)]
    new_col = max_col_index[:_small_round]
    return new_col

# Method 1: use duplicated() to check identical rows.
def check_duplicate_rows(df):
    # Find duplicate rows.
    duplicated_rows = df[df.duplicated()]
    
    if len(duplicated_rows) > 0:
        print(f"Found {len(duplicated_rows)} duplicate rows")
        print("\nDuplicate rows:")
        print(duplicated_rows)
        
        print("\nIndices of duplicate rows:")
        print(duplicated_rows.index.tolist())
        
        return True
    else:
        print("No duplicate rows found")
        return False

# Method 2: show all duplicate rows (including the first occurrence).
def check_duplicate_rows_with_first(df):
    # Find all duplicate rows (including the first occurrence).
    duplicated_mask = df.duplicated(keep=False)
    duplicated_rows = df[duplicated_mask]
    
    if len(duplicated_rows) > 0:
        print(f"Found {len(duplicated_rows)} duplicate rows")
        print("\nDuplicate rows (including the first occurrence):")
        print(duplicated_rows.sort_values(by=list(df.columns)))  # Sort for easier inspection.
        
        print("\nIndices of duplicate rows:")
        print(duplicated_rows.index.tolist())
        
        return True
    else:
        print("No duplicate rows found")
        return False


def get_round_columns(encode_round, rounds_per=3):
    return [f"round{i}" for i in range(encode_round * rounds_per)]


def active_to_round_channel(active_indices, rounds_per=3):
    return "&".join(
        f"{idx // rounds_per + 1}-{idx % rounds_per + 1}"
        for idx in active_indices
    )


def _passes_same_channel_filter(
    active_indices,
    rounds_per,
    same_round_del,
    max_same_channel_count=None,
):
    if max_same_channel_count is not None:
        channel_counts = [0] * rounds_per
        for idx in active_indices:
            channel = idx % rounds_per
            channel_counts[channel] += 1
            if channel_counts[channel] > max_same_channel_count:
                return False

    if same_round_del is None:
        return True

    for idx1, idx2 in itertools.combinations(sorted(active_indices), 2):
        same_channel = (idx2 - idx1) % rounds_per == 0
        if not same_channel:
            continue

        round_gap = (idx2 - idx1) // rounds_per
        if same_round_del is False or round_gap <= same_round_del:
            return False

    return True


def generate_mhd4_candidates(
    encode_round,
    rounds_per=3,
    same_round_del=2,
    code_weight=4,
    max_same_channel_count=2,
):
    """
    Generate a candidate pool of individual codes satisfying the basic MHD4 constraints.

    Constraints:
    1. Each code has exactly code_weight active bits.
    2. The active bits in each code come from code_weight distinct rounds.
    3. Active bits on the same channel within a code must satisfy the same_round_del gap.
    4. Any channel appears at most max_same_channel_count times within a code.
       Pass None to leave the occurrence count unrestricted.
    """
    if (
        max_same_channel_count is not None
        and (
            isinstance(max_same_channel_count, (bool, np.bool_))
            or not isinstance(max_same_channel_count, (int, np.integer))
            or max_same_channel_count < 1
        )
    ):
        raise ValueError("max_same_channel_count must be a positive integer or None")

    if code_weight > encode_round:
        return []

    active_codes = []
    for selected_rounds in itertools.combinations(range(encode_round), code_weight):
        for channels in itertools.product(range(rounds_per), repeat=code_weight):
            active_indices = tuple(
                round_idx * rounds_per + channel_idx
                for round_idx, channel_idx in zip(selected_rounds, channels)
            )
            if _passes_same_channel_filter(
                active_indices,
                rounds_per,
                same_round_del,
                max_same_channel_count,
            ):
                active_codes.append(active_indices)

    return active_codes


def _build_mhd4_triple_index(active_codes):
    triples_by_code = [
        tuple(itertools.combinations(active_indices, 3))
        for active_indices in active_codes
    ]
    codes_by_triple = {}
    for code_idx, triples in enumerate(triples_by_code):
        for triple in triples:
            codes_by_triple.setdefault(triple, []).append(code_idx)

    return triples_by_code, codes_by_triple


def _dynamic_min_degree_indices(
    active_codes,
    triples_by_code,
    codes_by_triple,
    random_ties=None,
):
    """Dynamically select the candidate with the lowest current conflict degree."""
    code_count = len(active_codes)
    if code_count == 0:
        return []

    if random_ties is None:
        random_ties = [0.0] * code_count

    remaining_degree = [
        sum(len(codes_by_triple[triple]) - 1 for triple in triples)
        for triples in triples_by_code
    ]
    active = bytearray([1]) * code_count
    heap = [
        (remaining_degree[idx], active_codes[idx], idx)
        for idx in range(code_count)
    ]
    heapq.heapify(heap)
    selected_indices = []

    while heap:
        while heap:
            degree, _, code_idx = heapq.heappop(heap)
            if active[code_idx] and degree == remaining_degree[code_idx]:
                break
        else:
            break

        tied_indices = [code_idx]
        while heap and heap[0][0] == degree:
            tied_degree, _, tied_idx = heapq.heappop(heap)
            if active[tied_idx] and tied_degree == remaining_degree[tied_idx]:
                tied_indices.append(tied_idx)

        def tie_key(idx):
            neighbor_degree_sum = sum(
                remaining_degree[neighbor_idx]
                for triple in triples_by_code[idx]
                for neighbor_idx in codes_by_triple[triple]
                if active[neighbor_idx] and neighbor_idx != idx
            )
            return (
                -neighbor_degree_sum,
                random_ties[idx],
                active_codes[idx],
            )

        selected_idx = min(tied_indices, key=tie_key)
        for tied_idx in tied_indices:
            if tied_idx != selected_idx:
                heapq.heappush(
                    heap,
                    (
                        remaining_degree[tied_idx],
                        active_codes[tied_idx],
                        tied_idx,
                    ),
                )

        selected_indices.append(selected_idx)
        removed_indices = {selected_idx}
        for triple in triples_by_code[selected_idx]:
            removed_indices.update(
                idx for idx in codes_by_triple[triple] if active[idx]
            )

        # Remove all selected candidates first, then update remaining degrees to
        # avoid counting candidates that will be deleted together.
        for removed_idx in removed_indices:
            active[removed_idx] = 0

        degree_drops = {}
        for removed_idx in removed_indices:
            for triple in triples_by_code[removed_idx]:
                for neighbor_idx in codes_by_triple[triple]:
                    if active[neighbor_idx]:
                        degree_drops[neighbor_idx] = (
                            degree_drops.get(neighbor_idx, 0) + 1
                        )

        for neighbor_idx, degree_drop in degree_drops.items():
            remaining_degree[neighbor_idx] -= degree_drop
            heapq.heappush(
                heap,
                (
                    remaining_degree[neighbor_idx],
                    active_codes[neighbor_idx],
                    neighbor_idx,
                ),
            )

    return selected_indices


def _improve_one_to_many(
    active_codes,
    triples_by_code,
    selected_indices,
    max_swaps,
):
    """Replace one code with at least two candidates; accept only capacity-increasing swaps."""
    selected_indices = list(selected_indices)
    if max_swaps <= 0:
        return selected_indices

    for _ in range(max_swaps):
        selected_set = set(selected_indices)
        triple_owner = {
            triple: code_idx
            for code_idx in selected_indices
            for triple in triples_by_code[code_idx]
        }
        candidates_by_blocker = {}

        for code_idx, triples in enumerate(triples_by_code):
            if code_idx in selected_set:
                continue

            blockers = {
                triple_owner[triple]
                for triple in triples
                if triple in triple_owner
            }
            if len(blockers) == 1:
                blocker_idx = next(iter(blockers))
                candidates_by_blocker.setdefault(blocker_idx, []).append(code_idx)

        best_exchange = None
        best_exchange_key = None
        for blocker_idx, candidate_indices in candidates_by_blocker.items():
            if len(candidate_indices) < 2:
                continue

            candidate_codes = [active_codes[idx] for idx in candidate_indices]
            candidate_triples, candidate_triple_index = _build_mhd4_triple_index(
                candidate_codes
            )
            replacement_positions = _dynamic_min_degree_indices(
                candidate_codes,
                candidate_triples,
                candidate_triple_index,
            )
            if len(replacement_positions) < 2:
                continue

            replacement_indices = tuple(
                candidate_indices[position] for position in replacement_positions
            )
            gain = len(replacement_indices) - 1
            exchange_key = (
                -gain,
                active_codes[blocker_idx],
                tuple(active_codes[idx] for idx in replacement_indices),
            )
            if best_exchange_key is None or exchange_key < best_exchange_key:
                best_exchange_key = exchange_key
                best_exchange = (blocker_idx, replacement_indices)

        if best_exchange is None:
            break

        blocker_idx, replacement_indices = best_exchange
        selected_indices.remove(blocker_idx)
        selected_indices.extend(replacement_indices)

    return selected_indices


def _is_better_solution(active_codes, candidate_indices, best_indices):
    if len(candidate_indices) != len(best_indices):
        return len(candidate_indices) > len(best_indices)

    candidate_key = tuple(sorted(active_codes[idx] for idx in candidate_indices))
    best_key = tuple(sorted(active_codes[idx] for idx in best_indices))
    return candidate_key < best_key


def _select_dynamic_mhd4_codes(
    active_codes,
    selection_restarts,
    random_seed,
    local_search_swaps,
):
    if len(active_codes) == 0:
        return []

    triples_by_code, codes_by_triple = _build_mhd4_triple_index(active_codes)
    best_indices = _dynamic_min_degree_indices(
        active_codes,
        triples_by_code,
        codes_by_triple,
    )

    rng = random.Random(random_seed)
    for _ in range(selection_restarts):
        random_ties = [rng.random() for _ in active_codes]
        selected_indices = _dynamic_min_degree_indices(
            active_codes,
            triples_by_code,
            codes_by_triple,
            random_ties=random_ties,
        )
        if _is_better_solution(active_codes, selected_indices, best_indices):
            best_indices = selected_indices

    # During restarts, compare only greedy results; run the more expensive local
    # search once on the best result at the end.
    best_indices = _improve_one_to_many(
        active_codes,
        triples_by_code,
        best_indices,
        local_search_swaps,
    )

    return [active_codes[idx] for idx in best_indices]


def _max_channel_count(active_indices, rounds_per):
    channel_counts = [0] * rounds_per
    for idx in active_indices:
        channel_counts[idx % rounds_per] += 1
    return max(channel_counts, default=0)


def _select_monotonic_channel_limits(
    active_codes,
    rounds_per,
    selection_restarts,
    random_seed,
    local_search_swaps,
):
    codes_by_limit = {}
    for active_indices in active_codes:
        channel_limit = _max_channel_count(active_indices, rounds_per)
        codes_by_limit.setdefault(channel_limit, []).append(active_indices)

    prefix_codes = []
    best_codes = []
    for channel_limit in sorted(codes_by_limit):
        new_codes = codes_by_limit[channel_limit]
        prefix_codes.extend(new_codes)

        fresh_codes = _select_dynamic_mhd4_codes(
            prefix_codes,
            selection_restarts,
            random_seed,
            local_search_swaps,
        )
        if len(best_codes) == 0:
            best_codes = fresh_codes
            continue

        used_triples = {
            triple
            for active_indices in best_codes
            for triple in itertools.combinations(active_indices, 3)
        }
        compatible_new_codes = [
            active_indices
            for active_indices in new_codes
            if all(
                triple not in used_triples
                for triple in itertools.combinations(active_indices, 3)
            )
        ]
        added_codes = _select_dynamic_mhd4_codes(
            compatible_new_codes,
            selection_restarts,
            random_seed,
            local_search_swaps,
        )
        carried_codes = best_codes + added_codes

        prefix_triples, _ = _build_mhd4_triple_index(prefix_codes)
        prefix_index = {code: idx for idx, code in enumerate(prefix_codes)}
        carried_indices = [prefix_index[code] for code in carried_codes]
        carried_indices = _improve_one_to_many(
            prefix_codes,
            prefix_triples,
            carried_indices,
            local_search_swaps,
        )
        carried_codes = [prefix_codes[idx] for idx in carried_indices]

        # When capacities tie, retain the inherited solution to keep codebooks
        # as stable as possible for adjacent N values.
        if len(fresh_codes) > len(carried_codes):
            best_codes = fresh_codes
        else:
            best_codes = carried_codes

    return best_codes


def select_pairwise_mhd4_codes(
    active_codes,
    code_weight=4,
    min_hamming_distance=4,
    selection_restarts=0,
    random_seed=1,
    rounds_per=None,
    local_search_swaps=20,
):
    """
    Select codes from the candidate pool so every pair has at least the given
    Hamming distance.

    For the MHD4 4-bit / min distance 4 case, no pair of code entries may share any
    3-bit subset. Use dynamic minimum-residual-conflict greedy selection and
    local gain swaps.

    When rounds_per is provided, process same-channel counts layer by layer so
    capacity does not decrease as the constraint is relaxed.
    """
    if min_hamming_distance is None or min_hamming_distance is False:
        return list(active_codes)

    if len(active_codes) == 0:
        return []

    if code_weight == 4 and min_hamming_distance == 4:
        normalized_codes = []
        for active_indices in active_codes:
            normalized_code = tuple(sorted(active_indices))
            if (
                len(normalized_code) != code_weight
                or len(set(normalized_code)) != code_weight
            ):
                raise ValueError(
                    "MHD4 candidates must contain 4 distinct valid positions"
                )
            normalized_codes.append(normalized_code)

        # For duplicate candidates with the same position set, keep only the
        # first occurrence.
        active_codes = list(dict.fromkeys(normalized_codes))
        if rounds_per is not None:
            return _select_monotonic_channel_limits(
                active_codes,
                rounds_per,
                selection_restarts,
                random_seed,
                local_search_swaps,
            )

        return _select_dynamic_mhd4_codes(
            active_codes,
            selection_restarts,
            random_seed,
            local_search_swaps,
        )

    selected_codes = []
    selected_sets = []
    for active_indices in active_codes:
        active_set = set(active_indices)
        keep = True
        for selected_set in selected_sets:
            overlap = len(active_set & selected_set)
            distance = code_weight + code_weight - 2 * overlap
            if distance < min_hamming_distance:
                keep = False
                break
        if keep:
            selected_codes.append(active_indices)
            selected_sets.append(active_set)

    return selected_codes


def active_codes_to_dataframe(
    active_codes,
    encode_round,
    rounds_per=3,
):
    columns = get_round_columns(encode_round, rounds_per)
    values = np.zeros((len(active_codes), encode_round * rounds_per), dtype=np.uint8)
    for row_idx, active_indices in enumerate(active_codes):
        values[row_idx, list(active_indices)] = 1

    codebook = pd.DataFrame(values, columns=columns, dtype=np.uint8)
    codebook["round_channel"] = [
        active_to_round_channel(active_indices, rounds_per)
        for active_indices in active_codes
    ]
    return codebook


def make_codebook(
    encode_round,
    rounds_per=3,
    same_round_del=2,
    code_weight=4,
    min_hamming_distance=4,
    selection_restarts=0,
    random_seed=1,
    include_round_channel=False,
    max_same_channel_count=2,
    local_search_swaps=20,
):
    """
    Main entry point for constrained MHD4 codebooks.

    The returned codebook has passed all MHD4 constraint filters, and shape[0]
    is the final available capacity.
    max_same_channel_count=N means any channel appears at most N times in each code.
    local_search_swaps is the maximum number of local gain swaps; set it to 0 to disable them.
    """
    active_codes = generate_mhd4_candidates(
        encode_round=encode_round,
        rounds_per=rounds_per,
        same_round_del=same_round_del,
        code_weight=code_weight,
        max_same_channel_count=max_same_channel_count,
    )
    selected_codes = select_pairwise_mhd4_codes(
        active_codes,
        code_weight=code_weight,
        min_hamming_distance=min_hamming_distance,
        selection_restarts=selection_restarts,
        random_seed=random_seed,
        rounds_per=rounds_per,
        local_search_swaps=local_search_swaps,
    )
    codebook = active_codes_to_dataframe(selected_codes, encode_round, rounds_per)

    if not include_round_channel:
        codebook = codebook.drop(columns=["round_channel"])

    return codebook


def min_pairwise_hamming_distance(codebook):
    values = codebook.select_dtypes(include=[np.number]).to_numpy(dtype=np.uint8)
    if values.shape[0] < 2:
        return None

    min_distance = values.shape[1]
    for row_idx in range(values.shape[0] - 1):
        distances = np.sum(values[row_idx + 1:] != values[row_idx], axis=1)
        min_distance = min(min_distance, int(distances.min()))

    return min_distance


def check_mhd4_codebook(
    codebook,
    rounds_per=3,
    code_weight=4,
    min_hamming_distance=4,
    same_round_del=2,
):
    values = codebook.select_dtypes(include=[np.number]).to_numpy(dtype=np.uint8)
    if values.shape[1] % rounds_per != 0:
        return False

    if not np.all(values.sum(axis=1) == code_weight):
        return False

    encode_round = values.shape[1] // rounds_per
    for round_idx in range(encode_round):
        start = round_idx * rounds_per
        end = start + rounds_per
        if np.any(values[:, start:end].sum(axis=1) > 1):
            return False

    for row in values:
        active_indices = np.flatnonzero(row)
        if not _passes_same_channel_filter(active_indices, rounds_per, same_round_del):
            return False

    min_distance = min_pairwise_hamming_distance(codebook)
    return min_distance is None or min_distance >= min_hamming_distance


def select_balanced_subset(
    codebook,
    need_num,
    max_restarts=100,
    candidate_pool_size=512,
    random_seed=1,
):
    """
    Select need_num rows from an MHD4-valid codebook and balance bit usage as
    much as possible.
    The subset inherits the original codebook's pairwise MHD4 distance constraint.
    """
    if need_num > codebook.shape[0]:
        raise ValueError(f"need_num={need_num} exceeds the available MHD4 capacity {codebook.shape[0]}")

    numeric_columns = codebook.select_dtypes(include=[np.number]).columns
    values = codebook[numeric_columns].to_numpy(dtype=np.uint8)
    target_sum = (need_num / codebook.shape[0]) * values.sum(axis=0)
    rng = np.random.RandomState(random_seed)

    best_selected = None
    best_score = float("inf")

    for _ in range(max_restarts):
        selected_positions = []
        selected_mask = np.zeros(codebook.shape[0], dtype=bool)
        current_sum = np.zeros(values.shape[1], dtype=float)

        while len(selected_positions) < need_num:
            available_positions = np.flatnonzero(~selected_mask)
            if len(available_positions) > candidate_pool_size:
                pool = rng.choice(
                    available_positions,
                    size=candidate_pool_size,
                    replace=False,
                )
            else:
                pool = available_positions

            new_sums = current_sum + values[pool]
            scores = np.abs(new_sums - target_sum).sum(axis=1)
            chosen_pos = int(pool[int(np.argmin(scores))])

            selected_positions.append(chosen_pos)
            selected_mask[chosen_pos] = True
            current_sum = current_sum + values[chosen_pos]

        score = np.abs(current_sum - target_sum).sum()
        if score < best_score:
            best_score = score
            best_selected = selected_positions

    return codebook.iloc[best_selected].copy()


def create_codebook(
    encode_round,
    need_num=None,
    rounds_per=3,
    same_round_del=2,
    code_weight=4,
    min_hamming_distance=4,
    selection_restarts=0,
    random_seed=1,
):
    codebook = make_codebook(
        encode_round=encode_round,
        rounds_per=rounds_per,
        same_round_del=same_round_del,
        code_weight=code_weight,
        min_hamming_distance=min_hamming_distance,
        selection_restarts=selection_restarts,
        random_seed=random_seed,
    )

    if need_num is None:
        return codebook

    return select_balanced_subset(
        codebook,
        need_num=need_num,
        random_seed=random_seed,
    )


def _numeric_codebook(codebook):
    return codebook.select_dtypes(include=[np.number])


def _aligned_expression(final_filtered_exp, codebook):
    if isinstance(final_filtered_exp, pd.DataFrame):
        if set(codebook.index).issubset(set(final_filtered_exp.columns)):
            return final_filtered_exp.loc[:, codebook.index]
        if final_filtered_exp.shape[1] == codebook.shape[0]:
            return final_filtered_exp
        raise ValueError("final_filtered_exp columns must align with codebook.index or equal the number of genes")
    return final_filtered_exp


def _to_2d_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _celltype_round_expression(codebook, final_filtered_exp):
    """Return a signal matrix with rows=cell types and columns=bits."""
    numeric_codebook = _numeric_codebook(codebook)
    aligned_exp = _aligned_expression(final_filtered_exp, numeric_codebook)
    values = np.asarray(aligned_exp, dtype=float) @ numeric_codebook.to_numpy(dtype=float)
    index = aligned_exp.index if isinstance(aligned_exp, pd.DataFrame) else pd.RangeIndex(values.shape[0])
    return pd.DataFrame(values, index=index, columns=numeric_codebook.columns)


def _abundance_vector(cell_abundance, celltype_index):
    """Align abundance to the expression matrix's cell-type order and normalize it to weights."""
    if isinstance(cell_abundance, pd.Series):
        values = cell_abundance.copy()
    elif isinstance(cell_abundance, pd.DataFrame):
        if cell_abundance.shape[1] != 1:
            raise ValueError("cell_abundance must have exactly one abundance column")
        values = cell_abundance.iloc[:, 0].copy()
    else:
        raw = np.asarray(cell_abundance).reshape(-1)
        if len(raw) != len(celltype_index):
            raise ValueError("the length of cell_abundance must equal the number of cell types")
        values = pd.Series(raw, index=celltype_index)

    values = pd.to_numeric(values, errors="raise").reindex(celltype_index)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise ValueError("cell_abundance is missing cell types: %s" % missing[:8])
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("cell_abundance must contain only finite values")
    total = float(values.sum())
    if (values < 0).any() or total <= 0:
        raise ValueError("cell_abundance must be non-negative and sum to more than 0")
    return values.astype(float) / total


def codebook_round_proportion(codebook, final_filtered_exp, cell_abundance):
    """Compute abundance-weighted within-cell-type bit signal proportion error; lower is more balanced.

    For each cell type, normalize bit signals to proportions and compare them
    with the ideal proportion ``1 / number of bits`` using squared error. Then
    weight each cell type's error by normalized cell abundance.
    """
    celltype_round_exp = _celltype_round_expression(codebook, final_filtered_exp)
    totals = celltype_round_exp.sum(axis=1)
    proportions = celltype_round_exp.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    target = 1.0 / celltype_round_exp.shape[1]
    per_type_error = ((proportions - target) ** 2).sum(axis=1)
    weights = _abundance_vector(cell_abundance, celltype_round_exp.index)
    return float(np.dot(per_type_error.to_numpy(), weights.to_numpy()))


def codebook_round_var(codebook, final_filtered_exp, cell_abundance):
    """
    Objective: compute the variance of total signal per round from the
    expression matrix and cell abundances.
    Lower variance indicates a more balanced overall load across rounds.
    """
    numeric_codebook = _numeric_codebook(codebook)
    aligned_exp = _aligned_expression(final_filtered_exp, numeric_codebook)

    expr = np.asarray(aligned_exp, dtype=float)
    code_values = numeric_codebook.to_numpy(dtype=float)
    abundance = _to_2d_array(cell_abundance)

    celltype_round_exp = np.dot(expr, code_values)
    round_exp = np.dot(celltype_round_exp.T, abundance)
    return float(np.var(round_exp[:, 0]))


def codebook_mt_var(codebook, final_filtered_exp):
    """
    Compute the overall variance of the celltype x round signal matrix.
    """
    numeric_codebook = _numeric_codebook(codebook)
    aligned_exp = _aligned_expression(final_filtered_exp, numeric_codebook)

    expr = np.asarray(aligned_exp, dtype=float)
    code_values = numeric_codebook.to_numpy(dtype=float)
    celltype_round_exp = np.dot(expr, code_values)
    return float(celltype_round_exp.reshape(-1).var())


def codebook_mt_max(codebook, final_filtered_exp):
    """
    Compute the maximum of the celltype x round signal matrix for the cut constraint.
    """
    numeric_codebook = _numeric_codebook(codebook)
    aligned_exp = _aligned_expression(final_filtered_exp, numeric_codebook)

    expr = np.asarray(aligned_exp, dtype=float)
    code_values = numeric_codebook.to_numpy(dtype=float)
    celltype_round_exp = np.dot(expr, code_values)
    return float(celltype_round_exp.reshape(-1).max())


def check_duplicates(codebook):
    return codebook.duplicated().any()


def _swap_rows_by_position(codebook, pos1, pos2):
    values = codebook.to_numpy(copy=True)
    values[[pos1, pos2]] = values[[pos2, pos1]]
    return pd.DataFrame(values, index=codebook.index, columns=codebook.columns)


def _swap_blocks_by_position(codebook, start1, start2, block_size):
    values = codebook.to_numpy(copy=True)
    block1 = values[start1:start1 + block_size].copy()
    block2 = values[start2:start2 + block_size].copy()
    values[start1:start1 + block_size] = block2
    values[start2:start2 + block_size] = block1
    return pd.DataFrame(values, index=codebook.index, columns=codebook.columns)


def generate_new_codebook_improved(codebook, method="random", max_attempts=100, rng=None):
    """
    Generate a new solution by swapping codebook rows.
    Only the gene-code assignment changes; the MHD4 codes themselves remain
    unchanged, so the MHD4 constraints are preserved.
    """
    if rng is None:
        rng = random.Random()

    if codebook.shape[0] < 2:
        return codebook.copy()

    for _ in range(max_attempts):
        new_codebook = codebook.copy()

        if method == "random":
            idx1 = rng.randrange(codebook.shape[0])
            idx2 = rng.randrange(codebook.shape[0])
            new_codebook = _swap_rows_by_position(new_codebook, idx1, idx2)

        elif method == "block_swap":
            block_size = rng.randint(2, max(2, min(5, codebook.shape[0] // 2)))
            start1 = rng.randrange(0, codebook.shape[0] - block_size + 1)
            start2 = rng.randrange(0, codebook.shape[0] - block_size + 1)
            new_codebook = _swap_blocks_by_position(new_codebook, start1, start2, block_size)

        elif method == "multi_swap":
            num_swaps = rng.randint(2, 4)
            for _ in range(num_swaps):
                idx1 = rng.randrange(codebook.shape[0])
                idx2 = rng.randrange(codebook.shape[0])
                new_codebook = _swap_rows_by_position(new_codebook, idx1, idx2)

        elif method == "reverse":
            start = rng.randrange(0, codebook.shape[0] - 1)
            end = rng.randrange(start + 1, codebook.shape[0])
            values = new_codebook.to_numpy(copy=True)
            values[start:end + 1] = values[start:end + 1][::-1]
            new_codebook = pd.DataFrame(values, index=codebook.index, columns=codebook.columns)

        else:
            raise ValueError(f"Unknown perturbation method: {method}")

        if not check_duplicates(new_codebook):
            return new_codebook

    return codebook.copy()


def metropolis_with_cut(old_codebook, new_codebook, t, final_filtered_exp, cell_abundance,
                        cut_value, rng=None, score_fn=codebook_round_var):
    """
    Metropolis acceptance criterion with a cut constraint.
    """
    if rng is None:
        rng = random.Random()

    old_score = score_fn(old_codebook, final_filtered_exp, cell_abundance)
    new_max = codebook_mt_max(new_codebook, final_filtered_exp)
    if new_max > cut_value:
        return old_codebook.copy(), old_score

    new_score = score_fn(new_codebook, final_filtered_exp, cell_abundance)
    delta = new_score - old_score

    if delta < 0:
        return new_codebook.copy(), new_score

    if t > 100 and old_score > 0:
        relative_delta = delta / old_score
        acceptance_prob = np.exp(-relative_delta / t) * 0.01
        if rng.random() < acceptance_prob:
            return new_codebook.copy(), new_score

    return old_codebook.copy(), old_score


def metropolis_with_cut_proportion(*args, **kwargs):
    """Use the Metropolis acceptance criterion with the proportion-error objective."""
    kwargs["score_fn"] = codebook_round_proportion
    return metropolis_with_cut(*args, **kwargs)


class _NullProgress:
    def __init__(self, total=None, desc=None):
        self.total = total
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value=1):
        return None


def _progress_bar(total=None, desc=None, use_tqdm=True):
    if not use_tqdm:
        return _NullProgress(total=total, desc=desc)
    try:
        import tqdm
        return tqdm(total=total, desc=desc)
    except Exception:
        return _NullProgress(total=total, desc=desc)


def plot_current_status(current_evolution, current_max_values, current_temp, cut_value,
                        restart_num, objective_label="Round Variance"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(current_evolution, "b-", alpha=0.6)
    plt.ylabel(objective_label)

    plt.subplot(132)
    plt.plot(current_max_values, "r-", alpha=0.6)
    plt.axhline(y=cut_value, color="g", linestyle="--", label="Cut line")
    plt.ylabel("Max info")
    plt.legend()

    plt.subplot(133)
    plt.plot(current_temp, "g-", alpha=0.6)
    plt.ylabel("Tm")
    plt.yscale("log")

    plt.suptitle(f"Big Iteration {restart_num + 1}")
    plt.tight_layout()
    plt.show()


def parallel_simulated_annealing_with_cut_and_viz(
    codebook,
    need_num,
    final_filtered_exp,
    cell_abundance,
    cut_value,
    save_path=None,
    num_restarts=5,
    _T0=1000,
    _Tend=10,
    _L=200,
    _q=0.9,
    max_attempts=100,
    plot_interval=50,
    random_seed=1,
    use_tqdm=True,
    log_every=0,
    score_fn=codebook_round_var,
    objective_label="Round Variance",
):
    """
    Optimize gene-codeword assignments by simulated annealing under MHD4 constraints.

    Parameters:
    - codebook: MHD4-valid code pool generated by make_codebook();
    - need_num: number of codes to assign to genes, usually equal to the number of final_filtered_exp.columns;
    - final_filtered_exp: expression matrix with rows=cell types and columns=genes;
    - cell_abundance: cell-type abundance vector;
    - cut_value: upper bound for the maximum of the celltype x round signal matrix.
    """
    if need_num > codebook.shape[0]:
        raise ValueError(f"need_num={need_num} exceeds the available MHD4 capacity {codebook.shape[0]}")

    codebook = _numeric_codebook(codebook).copy()

    if isinstance(final_filtered_exp, pd.DataFrame) and need_num != len(final_filtered_exp.columns):
        raise ValueError("need_num must equal the number of genes in final_filtered_exp.columns")

    rng = random.Random(random_seed)
    best_codebook = None
    best_score = float("inf")
    all_evolution = []
    all_max_values = []

    total_iterations = (int(np.log(_Tend / _T0) / np.log(_q)) + 1) * _L
    methods = ["random", "block_swap", "multi_swap", "reverse"]

    for restart in range(num_restarts):
        print(f"\nStarting restart {restart + 1}/{num_restarts}.")

        init_codebook = None
        init_max = None
        with _progress_bar(total=max_attempts, desc="Finding initial solution", use_tqdm=use_tqdm) as pbar:
            for attempt in range(max_attempts):
                init_codebook = select_balanced_subset(
                    codebook,
                    need_num,
                    random_seed=rng.randrange(1, 2 ** 31 - 1),
                )
                if isinstance(final_filtered_exp, pd.DataFrame):
                    init_codebook.index = final_filtered_exp.columns

                init_max = codebook_mt_max(init_codebook, final_filtered_exp)
                if init_max <= cut_value:
                    break

                pbar.update(1)
            else:
                print(f"Warning: restart {restart + 1} could not find an initial solution satisfying the cut constraint")
                continue

        t = _T0
        evolution = []
        max_values = []
        temp_history = []
        current_codebook = init_codebook
        current_score = score_fn(current_codebook, final_filtered_exp, cell_abundance)

        print(f"Initial state - maximum: {init_max:.2f}, objective: {current_score:.6g}")

        with _progress_bar(total=total_iterations, desc="Optimization progress", use_tqdm=use_tqdm) as pbar:
            iteration = 0
            while t > _Tend:
                for _ in range(_L):
                    method = rng.choice(methods)
                    new_codebook = generate_new_codebook_improved(current_codebook, method=method, rng=rng)
                    current_codebook, current_score = metropolis_with_cut(
                        current_codebook,
                        new_codebook,
                        t,
                        final_filtered_exp,
                        cell_abundance,
                        cut_value,
                        rng=rng,
                        score_fn=score_fn,
                    )

                    current_max = codebook_mt_max(current_codebook, final_filtered_exp)
                    evolution.append(current_score)
                    max_values.append(current_max)
                    temp_history.append(t)

                    if current_score < best_score and current_max <= cut_value:
                        best_score = current_score
                        best_codebook = current_codebook.copy()

                    iteration += 1
                    pbar.update(1)

                    # Log at each temperature step for runtime visibility,
                    # so it is clear that the objective is decreasing rather than idling.
                    if log_every and iteration % log_every == 0:
                        print("  Annealing restart %d/%d · iteration %d/%d · temperature %.1f · "
                              "current objective %.4g · best %.4g"
                              % (restart + 1, num_restarts, iteration, total_iterations,
                                 t, current_score, best_score), flush=True)

                    if plot_interval and iteration % plot_interval == 0:
                        plot_current_status(evolution, max_values, temp_history, cut_value,
                                            restart, objective_label=objective_label)

                t = t * _q

        all_evolution.append(evolution)
        all_max_values.append(max_values)

        final_max = codebook_mt_max(current_codebook, final_filtered_exp)
        print(f"\nRestart {restart + 1} complete:")
        print(f"- Final maximum: {final_max:.2f}")
        print(f"- Final objective: {current_score:.6g}")
        print(f"- Best objective so far: {best_score:.6g}")

    if save_path and all_evolution:
        try:
            import matplotlib.pyplot as plt

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(15, 10))
            plt.subplot(211)
            for idx, evolution in enumerate(all_evolution):
                plt.plot(evolution, alpha=0.5, label=f"Restart {idx + 1}")
            plt.xlabel("Iterations")
            plt.ylabel(objective_label)
            plt.legend()

            plt.subplot(212)
            for idx, max_values in enumerate(all_max_values):
                plt.plot(max_values, alpha=0.5, label=f"Restart {idx + 1}")
            plt.axhline(y=cut_value, color="r", linestyle="--", label="Cut limit")
            plt.xlabel("Iterations")
            plt.ylabel("Maximum Value")
            plt.legend()

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(f"Warning: could not save optimization plot: {exc}")

    if best_codebook is not None:
        final_best_max = codebook_mt_max(best_codebook, final_filtered_exp)
        print("\nFinal optimization result:")
        print(f"- Best solution maximum: {final_best_max:.2f}")
        print(f"- Best objective: {best_score:.6g}")
    else:
        print("\nWarning: no feasible solution found")

    return best_codebook, best_score, all_evolution


def parallel_simulated_annealing_with_cut_and_viz_proportion(*args, **kwargs):
    """Run simulated annealing with the cell-type proportion objective."""
    kwargs["score_fn"] = codebook_round_proportion
    kwargs["objective_label"] = "Proportion Error"
    return parallel_simulated_annealing_with_cut_and_viz(*args, **kwargs)


def check_same_channel_ones(df, n_channels=3):
    # Keep only encoded columns such as round0, round1, ... and sort numerically.
    round_cols = [
        col for col in df.columns
        if re.fullmatch(r"round\d+", str(col))
    ]
    round_cols = sorted(
        round_cols,
        key=lambda col: int(re.search(r"\d+$", str(col)).group()),
    )

    if len(round_cols) % n_channels != 0:
        raise ValueError(
            f"The number of round columns ({len(round_cols)}) is not divisible by {n_channels}."
        )

    values = df[round_cols].to_numpy()

    # Take every third column to count the number of 1s for each channel across rounds.
    channel_counts = np.column_stack([
        values[:, channel_idx::n_channels].sum(axis=1)
        for channel_idx in range(n_channels)
    ])

    result = df.copy()

    for channel_idx in range(n_channels):
        result[f"channel_{channel_idx + 1}_n1"] = (
            channel_counts[:, channel_idx]
        )

    # Maximum number of 1s in any one channel.
    result["max_same_channel_n1"] = channel_counts.max(axis=1)

    # Whether any channel is 1 in two or more rounds.
    result["has_repeated_same_channel"] = (
        result["max_same_channel_n1"] >= 2
    )

    return result
