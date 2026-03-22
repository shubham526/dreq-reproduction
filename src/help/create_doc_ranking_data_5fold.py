"""
create_doc_ranking_data_5fold_dreq.py

Creates fold-wise doc ranking data for 5-fold CV using the DREQ doc_ranking.
Loads shared inputs (docs, embeddings, encoder) ONCE and reuses them
across all 15 splits (5 folds x 3 splits).

Key differences from the QDER 5-fold script:
  - Imports from make_doc_ranking_data_dreq.py (not the QDER script)
  - Loads a neural encoder (BERT etc.) once and passes it to create_data
  - No entity_names or k args (DREQ sums entity embeddings, no query expansion)
  - Expects pre-chunked docs (run preprocess_docs.py first)

Usage:
    python create_doc_ranking_data_5fold_dreq.py \
        --fold-splits        /path/to/fold_splits \
        --entity-run-base    /path/to/entity_run_splits_dir \
        --output-base        /path/to/output_dir \
        --docs               /path/to/docs_chunked.jsonl \
        --embeddings         /path/to/mmead_entities.wikipedia2vec.jsonl.gz \
        --doc-run-type       bm25 \
        --encoder            bert \
        --max-len            512 \
        --entity-weighting   minmax \
        --use-cuda
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import shared logic from make_doc_ranking_data_dreq.py.
# Both scripts must live in the same directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAKE_DATA_SCRIPT = os.path.join(SCRIPT_DIR, "make_doc_ranking_data_dreq.py")

if not os.path.exists(MAKE_DATA_SCRIPT):
    print(f"[ERROR] make_doc_ranking_data_dreq.py not found at {MAKE_DATA_SCRIPT}")
    print("  Place this script in the same directory as make_doc_ranking_data_dreq.py.")
    sys.exit(1)

sys.path.insert(0, SCRIPT_DIR)
from make_doc_ranking_data import (
    Encoder,
    load_docs,
    load_embeddings,
    load_queries,
    read_qrels,
    read_run,
    create_data,
    WEIGHTING_SCHEMES,
)

MODEL_MAP = {
    'bert':      'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta':   'roberta-base',
    'deberta':   'microsoft/deberta-base',
    'ernie':     'nghuyong/ernie-2.0-base-en',
    'electra':   'google/electra-small-discriminator',
    'conv-bert': 'YituTech/conv-bert-base',
    't5':        't5-base',
}


# =============================================================================
#  HELPERS
# =============================================================================

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def log(msg):
    print(f"\n{'=' * 64}\n  {msg}\n{'=' * 64}")


def check_file(path):
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)


def check_dir(path):
    if not os.path.isdir(path):
        print(f"[ERROR] Directory not found: {path}")
        sys.exit(1)


def preflight_check(args):
    """Verify all required files and directories exist before starting."""
    log("Pre-flight checks")

    check_file(args.docs)
    check_file(args.embeddings)
    check_dir(args.fold_splits)
    check_dir(args.entity_run_base)
    check_dir(args.output_base)

    doc_run_subdir = f"{args.doc_run_type}_run_splits"
    doc_run_path = os.path.join(args.fold_splits, doc_run_subdir)
    if not os.path.isdir(doc_run_path):
        print(f"[ERROR] Doc run directory not found: {doc_run_path}")
        sys.exit(1)

    splits_needed = {
        doc_run_subdir:     ("training.run.txt",  "validation.run.txt",  "testing.run.txt"),
        "doc_qrels_splits": ("training.qrels.txt", "validation.qrels.txt", "testing.qrels.txt"),
        "queries_splits":   ("training.tsv",        "validation.tsv",       "testing.tsv"),
    }

    for fold in args.folds:
        fold_dir = f"fold-{fold}"

        for subdir, files in splits_needed.items():
            for fname in files:
                check_file(os.path.join(args.fold_splits, subdir, fold_dir, fname))

        for fname in ("training.run.txt", "validation.run.txt", "testing.run.txt"):
            check_file(os.path.join(args.entity_run_base, fold_dir, fname))

        check_dir(os.path.join(args.output_base, fold_dir))

    print("  All input files found.")
    print(f"  docs               : {args.docs}")
    print(f"  embeddings         : {args.embeddings}")
    print(f"  fold_splits        : {args.fold_splits}")
    print(f"  entity_runs        : {args.entity_run_base}")
    print(f"  output_base        : {args.output_base}")
    print(f"  doc_run_type       : {args.doc_run_type}")
    print(f"  encoder            : {args.encoder}")
    print(f"  max_len            : {args.max_len}")
    print(f"  folds              : {args.folds}")
    print(f"  splits             : {args.splits}")
    print(f"  balance            : {args.balance}")
    print(f"  entity_weighting   : {args.entity_weighting}")
    print(f"  filter             : {args.filter_no_entities}")
    print(f"  train_format       : {args.train_format}")
    print(f"  negatives_per_pos  : {args.negatives_per_pos}")
    print(f"  random_seed        : {args.random_seed}")
    print(f"  train_output       : {args.train_output_name or 'training.jsonl'}")
    print(f"  validation_output  : {args.validation_output_name or 'validation.jsonl'}")
    print(f"  test_output        : {args.test_output_name or 'testing.jsonl'}")


def count_lines(path):
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def print_split_stats(output_file, split_name):
    if os.path.isfile(output_file):
        lines = count_lines(output_file)
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"    {split_name}: {lines:,} examples  ({size_mb:.1f} MB)")
    else:
        print(f"    {split_name}: [NOT FOUND]")


def get_output_filename_for_split(split, args):
    if split == "training":
        return args.train_output_name or "training.jsonl"
    elif split == "validation":
        return args.validation_output_name or "validation.jsonl"
    elif split == "testing":
        return args.test_output_name or "testing.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create 5-fold CV DREQ doc ranking data with single shared input load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Shared inputs (loaded once)
    parser.add_argument("--docs", required=True,
                        help="Pre-chunked docs JSONL (run preprocess_docs.py first). "
                             "Each line must have: doc_id, entities, text, chunks.")
    parser.add_argument("--embeddings", required=True,
                        help="Wikipedia2Vec entity embeddings (.jsonl.gz)")

    # Encoder args (loaded once, shared across all folds)
    parser.add_argument("--encoder",
                        help="Encoder doc_ranking name (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). "
                             "Default: bert.",
                        choices=list(MODEL_MAP.keys()), default="bert", type=str)
    parser.add_argument("--max-len",
                        help="Maximum token length for truncation/padding. Default: 512",
                        default=512, type=int)
    parser.add_argument("--cuda",
                        help="CUDA device number. Default: 0.",
                        type=int, default=0)
    parser.add_argument("--use-cuda",
                        help="Use CUDA if available. Default: False.",
                        action="store_true")

    # Directory structure
    parser.add_argument("--fold-splits", required=True,
                        help="Root of fold splits dir, containing: "
                             "bm25_run_splits/, cedr_run_splits/, doc_qrels_splits/, queries_splits/")
    parser.add_argument("--entity-run-base", required=True,
                        help="Dir with entity run fold splits: fold-{0..4}/training.run.txt etc.")
    parser.add_argument("--output-base", required=True,
                        help="Root output dir with fold-{0..4}/ subdirs already created.")
    parser.add_argument("--train-output-name", default=None,
                        help="Output filename for training split inside each fold dir. "
                             "Default: training.jsonl")
    parser.add_argument("--validation-output-name", default=None,
                        help="Output filename for validation split inside each fold dir. "
                             "Default: validation.jsonl")
    parser.add_argument("--test-output-name", default=None,
                        help="Output filename for testing split inside each fold dir. "
                             "Default: testing.jsonl")
    parser.add_argument("--doc-run-type", choices=["bm25", "cedr"], default="bm25",
                        help="Which doc ranking run to use as baseline (default: bm25)")

    # Data creation options
    parser.add_argument("--balance", action="store_true",
                        help="Balance positive/negative examples in each split.")
    parser.add_argument("--entity-weighting",
                        choices=WEIGHTING_SCHEMES, default="minmax",
                        help=(
                            "How to weight entity scores before scaling embeddings.\n"
                            "  raw:            scores as-is.\n"
                            "  minmax:         per-query min-max normalisation to [0,1] — recommended.\n"
                            "  reciprocal:     1/rank weighting.\n"
                            "  uniform:        all weights = 1.0 — ablation baseline.\n"
                            "  log_reciprocal: 1/log2(rank+1) NDCG-style discount.\n"
                            "Default: minmax"
                        ))
    parser.add_argument("--filter-no-entities", action="store_true",
                        help="Filter docs with no entity overlap (old behaviour). "
                             "Default: False — keep all docs and use zero entity features.")
    parser.add_argument("--train-format",
                        choices=["pointwise", "pairwise"], default="pointwise", type=str,
                        help="Format for training data. Default: pointwise")
    parser.add_argument("--negatives-per-pos",
                        default=1, type=int,
                        help="Negative docs to sample per positive (pairwise only). Default: 1")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")

    # Selective execution
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Which folds to process (default: all 5)")
    parser.add_argument("--splits", nargs="+",
                        default=["training", "validation", "testing"],
                        choices=["training", "validation", "testing"],
                        help="Which splits to create (default: all 3)")

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # -------------------------------------------------------------------------
    # Pre-flight checks
    # -------------------------------------------------------------------------
    preflight_check(args)

    total_start = time.time()

    # -------------------------------------------------------------------------
    # Load shared inputs ONCE
    # -------------------------------------------------------------------------
    log("Loading shared inputs (docs, embeddings, encoder) — done once for all folds")

    t0 = time.time()
    print("  Loading documents (pre-chunked)...")
    docs = load_docs(args.docs)
    print(f"  Loaded {len(docs):,} documents in {fmt_time(time.time() - t0)}")

    t0 = time.time()
    print("  Loading embeddings...")
    embeddings = load_embeddings(args.embeddings)
    print(f"  Loaded {len(embeddings):,} embeddings in {fmt_time(time.time() - t0)}")

    t0 = time.time()
    print(f"  Loading encoder ({args.encoder})...")
    from transformers import AutoTokenizer
    cuda_device = f"cuda:{args.cuda}"
    device = torch.device(cuda_device if torch.cuda.is_available() and args.use_cuda else "cpu")
    pretrain = MODEL_MAP[args.encoder]
    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)
    encoder = Encoder(pretrained=pretrain)
    encoder.to(device)
    encoder.eval()
    print(f"  Encoder loaded in {fmt_time(time.time() - t0)} — device: {device}")

    # -------------------------------------------------------------------------
    # Process each fold
    # -------------------------------------------------------------------------
    grand_total_examples = 0
    all_stats = {}

    doc_run_subdir = f"{args.doc_run_type}_run_splits"

    for fold in args.folds:
        fold_dir = f"fold-{fold}"
        log(f"FOLD {fold}")

        fold_start = time.time()
        fold_total = 0
        all_stats[fold] = {}

        doc_fold     = os.path.join(args.fold_splits, doc_run_subdir,        fold_dir)
        qrels_fold   = os.path.join(args.fold_splits, "doc_qrels_splits",    fold_dir)
        queries_fold = os.path.join(args.fold_splits, "queries_splits",      fold_dir)
        entity_fold  = os.path.join(args.entity_run_base,                    fold_dir)
        out_fold     = os.path.join(args.output_base,                        fold_dir)

        for split in args.splits:
            split_start = time.time()

            is_train = split == "training"

            queries_file = os.path.join(queries_fold, f"{split}.tsv")
            qrels_file   = os.path.join(qrels_fold,   f"{split}.qrels.txt")
            doc_run_file = os.path.join(doc_fold,     f"{split}.run.txt")
            ent_run_file = os.path.join(entity_fold,  f"{split}.run.txt")
            output_name  = get_output_filename_for_split(split, args)
            output_file  = os.path.join(out_fold, output_name)

            print(f"\n  [{fold_dir}] {split.upper()}")
            print(f"    queries        : {queries_file}")
            print(f"    qrels          : {qrels_file}")
            print(f"    doc_run        : {doc_run_file}")
            print(f"    entity_run     : {ent_run_file}")
            print(f"    output         : {output_file}")
            print(f"    train          : {is_train}")
            print(f"    train_format   : {args.train_format if is_train else 'pointwise (eval)'}")

            queries = load_queries(queries_file)
            qrels   = read_qrels(qrels_file)
            doc_run = read_run(doc_run_file)
            ent_run = read_run(ent_run_file)

            # Clear output file before writing
            open(output_file, 'w').close()

            stats = create_data(
                queries=queries,
                docs=docs,
                doc_qrels=qrels,
                doc_run=doc_run,
                entity_run=ent_run,
                entity_embeddings=embeddings,
                train=is_train,
                balance=args.balance,
                save=output_file,
                encoder=encoder,
                tokenizer=tokenizer,
                max_len=args.max_len,
                device=device,
                filter_no_entities=args.filter_no_entities,
                entity_weighting=args.entity_weighting,
                train_format=args.train_format,
                negatives_per_pos=args.negatives_per_pos,
            )

            all_stats[fold][split] = stats
            elapsed = time.time() - split_start
            n_examples = stats.get('total_examples', 0)
            fold_total += n_examples

            print(f"    Done in {fmt_time(elapsed)} — {n_examples:,} examples")
            print_split_stats(output_file, split)

        fold_elapsed = time.time() - fold_start
        grand_total_examples += fold_total
        print(f"\n  [{fold_dir}] Total: {fold_total:,} examples in {fmt_time(fold_elapsed)}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    log("ALL FOLDS COMPLETE")
    print(f"  Encoder            : {args.encoder} ({pretrain})")
    print(f"  Device             : {device}")
    print(f"  Doc run type       : {args.doc_run_type}")
    print(f"  Entity weighting   : {args.entity_weighting}")
    print(f"  Train format       : {args.train_format}")
    print(f"  Negatives per pos  : {args.negatives_per_pos}")
    print(f"  Total examples     : {grand_total_examples:,}")
    print(f"  Total time         : {fmt_time(total_elapsed)}")
    print()
    print("  Per-fold breakdown:")

    for fold in args.folds:
        fold_dir = f"fold-{fold}"
        out_fold = os.path.join(args.output_base, fold_dir)
        print(f"\n  {fold_dir}:")
        fold_total = 0
        for split in ["training", "validation", "testing"]:
            output_name = get_output_filename_for_split(split, args)
            f = os.path.join(out_fold, output_name)
            n = count_lines(f)
            fold_total += n
            size_mb = os.path.getsize(f) / (1024 * 1024) if os.path.isfile(f) else 0
            print(f"    {split:<12}: {n:>8,} examples  ({size_mb:.1f} MB)  [{output_name}]")
        print(f"    {'total':<12}: {fold_total:>8,} examples")


if __name__ == "__main__":
    main()