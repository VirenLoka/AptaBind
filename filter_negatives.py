"""
filter_dkk1_negatives.py
------------------------
Filters floor-score sequences from a DKK1 SELEX enrichment file to produce
a clean set of high-confidence negatives for model training.

Usage:
    python filter_dkk1_negatives.py --input dkk1_scores.tsv --output dkk1_negatives_clean.tsv

Input format (TSV, no header):
    SEQUENCE<tab>ENRICHMENT_SCORE

All filters are documented with their biological rationale.
"""

import argparse
import re
import sys
import os
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# KNOWN DKK1 HIGH-AFFINITY APTAMER SEQUENCES
# From Figure 6E of the DeepAptamer paper.
# Any floor sequence too similar to these is
# a suspected binder and must be excluded.
# ─────────────────────────────────────────────
KNOWN_DKK1_BINDERS = [
    "TGTAGATAGAGGGGTTGGTGTGGTTGGCGCTATCC",   # AptDK-1,  Kd = 5.50 nM
    "TGTAGATAGAGGTTTTTTGTTTTTTTCGCTATCC",    # AptDK-1_mut, Kd = 13.10 nM
    "TCCTCTTCCTTTGGCTAGGTTGGTTAAGTTGGTGC",   # AptDK-2,  Kd = 1.97 nM
]

# ─────────────────────────────────────────────
# DKK1 BINDING MOTIFS (paper Figure 6D)
# The core motif is GGTTGG-NNN-GGTTGG.
# We also flag partial matches conservatively.
# ─────────────────────────────────────────────
DKK1_MOTIFS = [
    # Full core motif with 2-5 nt spacer (flexible around the NNN)
    r"GGTTGG.{2,5}GGTTGG",
    # Partial: single GGTTGG occurrence (more conservative flag)
    r"GGTTGG",
    # G-quadruplex-like runs (common in G-rich aptamers targeting DKK1)
    r"GGG.{1,7}GGG.{1,7}GGG",
    # Reverse complement of core motif: CCAACC-NNN-CCAACC
    r"CCAACC.{2,5}CCAACC",
]

# ─────────────────────────────────────────────
# VALID BASES — sequences with anything outside
# these are structurally ambiguous and excluded.
# ─────────────────────────────────────────────
VALID_BASES = set("ATCG")

# ─────────────────────────────────────────────
# EXPECTED SEQUENCE LENGTH
# The DeepAptamer library uses a 35-nt random
# region. Sequences outside this window are
# likely alignment artifacts.
# ─────────────────────────────────────────────
EXPECTED_MIN_LEN = 30
EXPECTED_MAX_LEN = 40


def load_data(filepath: str) -> list[tuple[str, float]]:
    """Load TSV file of (sequence, score) pairs."""
    records = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                log.warning(f"Line {lineno}: expected 2 columns, got {len(parts)} — skipping")
                continue
            seq, score_str = parts
            try:
                score = float(score_str)
            except ValueError:
                log.warning(f"Line {lineno}: cannot parse score '{score_str}' — skipping")
                continue
            records.append((seq.upper().strip(), score))
    log.info(f"Loaded {len(records):,} records from {filepath}")
    return records


def identify_floor_value(records: list[tuple[str, float]]) -> float:
    """
    Identify the floor enrichment value.
    The floor is the minimum score in the dataset — sequences assigned
    this value hit the lower detection limit of the scoring tool and
    could not be individually ranked.
    """
    scores = [s for _, s in records]
    floor = min(scores)
    count = sum(1 for s in scores if s == floor)
    log.info(f"Floor value: {floor:.6e}  |  Sequences at floor: {count:,}")
    return floor


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute edit distance between two sequences.
    Used to detect near-identical matches to known binders.
    Pure Python — no dependencies required.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,   # deletion
                curr[j] + 1,        # insertion
                prev[j] + (c1 != c2)  # substitution
            ))
        prev = curr
    return prev[-1]


def hamming_distance(s1: str, s2: str) -> int:
    """
    Hamming distance for equal-length sequences.
    Faster than Levenshtein for same-length comparison.
    """
    if len(s1) != len(s2):
        return None
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# ─────────────────────────────────────────────────────────────────────────────
# FILTER FUNCTIONS
# Each returns (passes: bool, reason: str)
# reason is only populated when the sequence fails.
# ─────────────────────────────────────────────────────────────────────────────

def filter_ambiguous_bases(seq: str) -> tuple[bool, str]:
    """
    FILTER 1: Ambiguous bases
    Rationale: N, R, Y, W, S etc. indicate sequencer uncertainty.
    DNAshape cannot compute structural features for ambiguous positions,
    and one-hot encoding is undefined for them. Hard exclude.
    """
    invalid = set(seq) - VALID_BASES
    if invalid:
        return False, f"ambiguous_bases:{','.join(sorted(invalid))}"
    return True, ""


def filter_length(seq: str) -> tuple[bool, str]:
    """
    FILTER 2: Sequence length
    Rationale: The DeepAptamer library has a 35-nt random region.
    Very short or very long sequences are likely alignment artifacts
    or chimeric reads from PCR errors.
    """
    if len(seq) < EXPECTED_MIN_LEN:
        return False, f"too_short:{len(seq)}"
    if len(seq) > EXPECTED_MAX_LEN:
        return False, f"too_long:{len(seq)}"
    return True, ""


def filter_low_complexity(seq: str) -> tuple[bool, str]:
    """
    FILTER 3: Low-complexity / homopolymer sequences
    Rationale: Sequences dominated by a single nucleotide (e.g. AAAAAAA...)
    are synthesis artifacts. They also cannot form stable secondary structures
    and are biologically implausible aptamer candidates. However we flag
    these as negatives cautiously — they are uninformative negatives because
    the model learns nothing useful from them. We exclude rather than include.
    Threshold: any single base > 70% of sequence.
    """
    for base in VALID_BASES:
        freq = seq.count(base) / len(seq)
        if freq > 0.70:
            return False, f"low_complexity:{base}={freq:.2f}"
    return True, ""


def filter_dkk1_motifs(seq: str) -> tuple[bool, str]:
    """
    FILTER 4: DKK1 binding motifs
    Rationale: Sequences containing the experimentally validated DKK1
    core motif (GGTTGG-NNN-GGTTGG) or its variants must not be labeled
    negative — they are likely true binders regardless of enrichment score.
    This is the most important filter given the paper's own finding that
    low-enrichment sequences can have Kd in the low nM range (AptDK-5,
    Kd = 2.55 nM).
    """
    for pattern in DKK1_MOTIFS:
        match = re.search(pattern, seq)
        if match:
            return False, f"contains_dkk1_motif:{pattern}@{match.start()}"
    return True, ""


def filter_known_binder_similarity(
    seq: str,
    max_hamming: int = 4,
    max_levenshtein: int = 5
) -> tuple[bool, str]:
    """
    FILTER 5: Similarity to known DKK1 binders
    Rationale: A sequence that differs from a confirmed binder by only a few
    bases is likely still a binder (especially if the mutations are outside
    the core motif). We exclude sequences within max_hamming substitutions
    of any known binder (same length) or within max_levenshtein edits
    (different length). Conservative thresholds — the paper's motif mutations
    showed that even 1-2 core residue changes can shift Kd from 5.5 to 13 nM,
    but non-core mutations had no effect. We use 4 Hamming as a safe buffer.
    """
    for binder in KNOWN_DKK1_BINDERS:
        if len(seq) == len(binder):
            hd = hamming_distance(seq, binder)
            if hd <= max_hamming:
                return False, f"similar_to_known_binder:hamming={hd}"
        else:
            ed = levenshtein_distance(seq, binder)
            if ed <= max_levenshtein:
                return False, f"similar_to_known_binder:levenshtein={ed}"
    return True, ""


def filter_gc_extremes(seq: str) -> tuple[bool, str]:
    """
    FILTER 6: Extreme GC content
    Rationale: Sequences with very high GC (>80%) tend to form stable
    non-specific secondary structures and aggregates. Sequences with very
    low GC (<20%) are unlikely to fold into functional 3D structures.
    Neither extreme is useful as a negative — the model should learn from
    sequences that are plausible aptamer candidates but don't bind DKK1,
    not from sequences that are structurally degenerate for unrelated reasons.
    """
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    if gc > 0.80:
        return False, f"high_gc:{gc:.2f}"
    if gc < 0.20:
        return False, f"low_gc:{gc:.2f}"
    return True, ""


def filter_duplicates(sequences: list[str]) -> set[str]:
    """
    FILTER 7: Exact duplicates
    Rationale: The same sequence appearing multiple times in the floor
    should only appear once in the negative set. Multiple copies could
    artificially inflate the weight of that sequence during training.
    Returns the set of unique sequences.
    """
    seen = set()
    dupes = 0
    unique = []
    for seq in sequences:
        if seq in seen:
            dupes += 1
        else:
            seen.add(seq)
            unique.append(seq)
    if dupes:
        log.info(f"  Removed {dupes:,} exact duplicates")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_filters(
    records: list[tuple[str, float]],
    floor_value: float,
    max_hamming: int = 4,
    max_levenshtein: int = 5,
) -> tuple[list[str], dict]:
    """
    Apply all filters to floor-value sequences.
    Returns (clean_sequences, filter_report).
    """

    # Step 0: extract only floor sequences
    floor_seqs = [seq for seq, score in records if score == floor_value]
    log.info(f"\n{'='*60}")
    log.info(f"Starting with {len(floor_seqs):,} floor-value sequences")
    log.info(f"{'='*60}")

    report = defaultdict(int)
    report["00_floor_total"] = len(floor_seqs)

    # Step 1: duplicates (before other filters to save time)
    floor_seqs = filter_duplicates(floor_seqs)
    report["01_after_dedup"] = len(floor_seqs)

    # Steps 2-7: per-sequence filters
    filters = [
        ("02_ambiguous_bases",          filter_ambiguous_bases),
        ("03_length",                   filter_length),
        ("04_low_complexity",           filter_low_complexity),
        ("05_dkk1_motifs",              filter_dkk1_motifs),
        ("06_gc_extremes",              filter_gc_extremes),
    ]

    passed = floor_seqs
    for stage_name, filter_fn in filters:
        before = len(passed)
        survivors = []
        removed = 0
        removal_reasons = defaultdict(int)

        for seq in passed:
            ok, reason = filter_fn(seq)
            if ok:
                survivors.append(seq)
            else:
                removed += 1
                # Bucket reason by its prefix (e.g. "ambiguous_bases")
                bucket = reason.split(":")[0]
                removal_reasons[bucket] += 1

        passed = survivors
        report[stage_name] = len(passed)
        log.info(
            f"[{stage_name}] {before:,} → {len(passed):,} "
            f"(removed {removed:,})"
        )
        for reason, count in sorted(removal_reasons.items()):
            log.info(f"    └─ {reason}: {count:,}")

    # Known binder similarity (separate because it has parameters)
    before = len(passed)
    survivors = []
    removed = 0
    for seq in passed:
        ok, reason = filter_known_binder_similarity(
            seq, max_hamming, max_levenshtein
        )
        if ok:
            survivors.append(seq)
        else:
            removed += 1
    passed = survivors
    report["07_known_binder_similarity"] = len(passed)
    log.info(
        f"[07_known_binder_similarity] {before:,} → {len(passed):,} "
        f"(removed {removed:,})"
    )

    report["08_final_clean_negatives"] = len(passed)
    return passed, dict(report)


def print_report(report: dict, floor_value: float) -> None:
    log.info(f"\n{'='*60}")
    log.info("FILTER REPORT SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Floor enrichment value : {floor_value:.6e}")
    log.info(f"{'─'*60}")
    prev = None
    for stage, count in sorted(report.items()):
        if prev is not None:
            removed = prev - count
            pct = removed / report["00_floor_total"] * 100 if report["00_floor_total"] else 0
            log.info(f"  {stage:<40} {count:>8,}   (-{removed:,}, {pct:.1f}%)")
        else:
            log.info(f"  {stage:<40} {count:>8,}")
        prev = count
    log.info(f"{'─'*60}")
    total_removed = report["00_floor_total"] - report["08_final_clean_negatives"]
    retention = report["08_final_clean_negatives"] / report["00_floor_total"] * 100
    log.info(f"  Total removed : {total_removed:,}")
    log.info(f"  Retention rate: {retention:.1f}%")
    log.info(f"  FINAL CLEAN NEGATIVES: {report['08_final_clean_negatives']:,}")
    log.info(f"{'='*60}\n")


def save_output(sequences: list[str], output_path: str, floor_value: float) -> None:
    with open(output_path, "w") as f:
        f.write(f"# DKK1 clean floor negatives\n")
        f.write(f"# Floor enrichment value: {floor_value:.6e}\n")
        f.write(f"# Total sequences: {len(sequences)}\n")
        f.write(f"# Filters applied: ambiguous_bases, length, low_complexity,\n")
        f.write(f"#   dkk1_motifs, gc_extremes, known_binder_similarity, duplicates\n")
        f.write(f"# Label for training: [0, 1] (negative / no binding)\n")
        f.write("sequence\tlabel\tenrichment_score\n")
        for seq in sequences:
            f.write(f"{seq}\t0\t{floor_value:.6e}\n")
    log.info(f"Saved {len(sequences):,} clean negatives → {output_path}")


def save_filter_log(report: dict, floor_value: float, log_path: str) -> None:
    with open(log_path, "w") as f:
        f.write("stage\tsequences_remaining\n")
        f.write(f"floor_value\t{floor_value:.6e}\n")
        for stage, count in sorted(report.items()):
            f.write(f"{stage}\t{count}\n")
    log.info(f"Filter log saved → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter DKK1 floor-score sequences to produce clean negatives."
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Input TSV file: sequence<tab>score (no header required)"
    )
    p.add_argument(
        "--output", "-o", default="dkk1_negatives_clean.tsv",
        help="Output TSV file for clean negatives (default: dkk1_negatives_clean.tsv)"
    )
    p.add_argument(
        "--log-file", default="filter_report.tsv",
        help="Output TSV log of per-stage counts (default: filter_report.tsv)"
    )
    p.add_argument(
        "--max-hamming", type=int, default=4,
        help="Max Hamming distance to known binders for same-length seqs (default: 4)"
    )
    p.add_argument(
        "--max-levenshtein", type=int, default=5,
        help="Max edit distance to known binders for diff-length seqs (default: 5)"
    )
    p.add_argument(
        "--floor-override", type=float, default=None,
        help="Manually specify floor value instead of auto-detecting minimum"
    )
    p.add_argument(
        "--motif-only", action="store_true",
        help="Run only the DKK1 motif filter (fast mode for inspection)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    records = load_data(args.input)
    if not records:
        log.error("No valid records loaded. Check file format.")
        sys.exit(1)

    floor_value = args.floor_override if args.floor_override else identify_floor_value(records)

    if args.motif_only:
        # Fast inspection mode: just check motifs in the whole file
        log.info("Running motif-only inspection on all sequences...")
        motif_hits = []
        for seq, score in records:
            ok, reason = filter_dkk1_motifs(seq)
            if not ok:
                motif_hits.append((seq, score, reason))
        log.info(f"Motif hits in full dataset: {len(motif_hits):,}")
        for seq, score, reason in motif_hits[:20]:
            log.info(f"  {seq}  score={score:.3e}  reason={reason}")
        return

    clean_seqs, report = run_filters(
        records,
        floor_value,
        max_hamming=args.max_hamming,
        max_levenshtein=args.max_levenshtein,
    )

    print_report(report, floor_value)
    save_output(clean_seqs, args.output, floor_value)
    save_filter_log(report, floor_value, args.log_file)

    if report["08_final_clean_negatives"] < 1000:
        log.warning(
            f"Only {report['08_final_clean_negatives']:,} clean negatives remain. "
            f"Consider supplementing with cross-target positives (CTGF/BCMA) "
            f"as additional Tier 1 negatives."
        )


if __name__ == "__main__":
    main()