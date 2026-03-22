"""
make_doc_ranking_data_dreq.py

Creates pointwise or pairwise JSONL training/test data for DREQ.

Requires documents to have been pre-processed by precompute_chunk_embs.py,
which handles spaCy chunking + BERT encoding offline. This script only handles:
  - Loading precomputed doc embeddings from .npy + metadata
  - Entity embedding weighting
  - Positive/negative document selection
  - JSONL serialisation

Usage:
    python make_doc_ranking_data_dreq.py \
        --queries   /path/to/queries.tsv \
        --doc-embs  /path/to/precomputed/bert_spacy10s5 \
        --qrels     /path/to/qrels.txt \
        --doc-run   /path/to/doc.run \
        --entity-run /path/to/entity.run \
        --embeddings /path/to/wikipedia2vec.jsonl.gz \
        --save      /path/to/output.jsonl \
        --train
"""

import json
import sys
import math
import gzip
import random
import argparse
import collections

import numpy as np
from tqdm import tqdm


# =============================================================================
#  FILE I/O HELPERS
# =============================================================================

def load_docs(precomp_dir: str):
    """
    Load precomputed doc embeddings produced by precompute_chunk_embs.py.

    Reads:
      <precomp_dir>/doc_embs.npy          float16/float32  [N, D]
      <precomp_dir>/doc_meta.jsonl.gz     {doc_id, entities, row_idx}
      <precomp_dir>/doc_id_to_row.json    {doc_id: row_idx}

    Returns:
      docs: {doc_id: (entities, doc_chunk_emb_list)}
        where doc_chunk_emb_list is a Python list of floats (float32), length D.
    """
    import os

    embs_path   = os.path.join(precomp_dir, 'doc_embs.npy')
    meta_path   = os.path.join(precomp_dir, 'doc_meta.jsonl.gz')
    id2row_path = os.path.join(precomp_dir, 'doc_id_to_row.json')

    for p in (embs_path, meta_path, id2row_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Precomputed file not found: {p}\n"
                "Run precompute_chunk_embs.py first."
            )

    print(f"Loading precomputed embeddings from {precomp_dir}...")
    embs = np.load(embs_path).astype(np.float32)   # always work in float32
    print(f"  Loaded embeddings: {embs.shape}")

    with open(id2row_path, 'r') as f:
        doc_id_to_row = json.load(f)

    docs = {}
    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading doc metadata"):
            d = json.loads(line)
            doc_id   = d['doc_id']
            entities = d['entities']
            row_idx  = d['row_idx']
            docs[doc_id] = (entities, embs[row_idx].tolist())

    print(f"  Loaded {len(docs):,} documents.")
    return docs


def read_qrels(qrels_file: str):
    qrels = collections.defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, object_id, relevance = line.strip().split()
            assert object_id not in qrels[query_id]
            qrels[query_id][object_id] = int(relevance)
    return qrels


def read_run(run_file: str):
    run = collections.defaultdict(dict)
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id, object_id, score = parts[0], parts[2], parts[4]
            if object_id not in run[query_id]:
                run[query_id][object_id] = float(score)
    return run


def load_embeddings(embedding_file: str):
    emb = {}
    print("Loading entity embeddings...")
    with gzip.open(embedding_file, 'r') as f:
        for line in tqdm(f, total=13032425, desc="Loading entity embeddings"):
            d = json.loads(line)
            emb[d['entity_id']] = d['embedding']
    print(f"  Loaded {len(emb):,} entity embeddings.")
    return emb


def load_queries(queries_file: str):
    with open(queries_file, 'r') as f:
        return {
            line.strip().split('\t')[0]: line.strip().split('\t')[1]
            for line in f
        }


def write_to_file(data_line: str, save: str):
    with open(save, 'a') as f:
        f.write(data_line + '\n')


# =============================================================================
#  ENTITY SCORE WEIGHTING
# =============================================================================

WEIGHTING_SCHEMES = ['raw', 'minmax', 'reciprocal', 'uniform', 'log_reciprocal']


def weight_entity_scores(entity_scores: dict, scheme: str) -> dict:
    """Re-weight a single query's entity scores according to the chosen scheme."""
    if not entity_scores:
        return entity_scores

    if scheme == 'raw':
        if any(s < 0 for s in entity_scores.values()):
            print("Warning: raw weighting with negative scores — "
                  "negative weights will flip embedding directions. Consider 'minmax'.")
        return dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True))

    if scheme == 'uniform':
        return {eid: 1.0 for eid in entity_scores}

    if scheme in ('reciprocal', 'log_reciprocal'):
        ranked = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            eid: (1.0 / rank if scheme == 'reciprocal' else 1.0 / math.log2(rank + 1))
            for rank, (eid, _) in enumerate(ranked, start=1)
        }

    if scheme == 'minmax':
        scores = list(entity_scores.values())
        min_s, max_s = min(scores), max(scores)
        if max_s > min_s:
            normalized = {eid: (s - min_s) / (max_s - min_s) for eid, s in entity_scores.items()}
        else:
            normalized = {eid: 1.0 for eid in entity_scores}
        return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))

    raise ValueError(f"Unknown weighting scheme: '{scheme}'. Choose from: {WEIGHTING_SCHEMES}")


# =============================================================================
#  EMBEDDING HELPERS
# =============================================================================

def get_query_entity_embeddings(query_entities: dict, entity_embeddings: dict) -> dict:
    """Subset the global embedding table to only entities in this query."""
    return {eid: entity_embeddings[eid] for eid in query_entities if eid in entity_embeddings}


def get_entity_centric_doc_embedding(doc_entities, query_entities, query_entity_embeddings):
    """
    DREQ-style: weighted sum of entity embeddings for entities shared between
    the document and query. Returns a [300] float32 list, or None if no overlap.
    """
    embeddings = []
    for entity_id in doc_entities:
        entity_id = str(entity_id)
        if entity_id in query_entity_embeddings and entity_id in query_entities:
            entity_embedding = query_entity_embeddings[entity_id]
            if len(entity_embedding) >= 300:
                weight = query_entities[entity_id]
                embeddings.append(weight * np.array(entity_embedding[:300]))

    if not embeddings:
        return None

    return np.sum(embeddings, axis=0, dtype=np.float32).tolist()


# =============================================================================
#  DOCUMENT COLLECTION
# =============================================================================

def get_docs(docs, qrels, query_entities, query_entity_embeddings,
             positive, query_docs, doc_run, filter_no_entities=False):
    """
    Collect positive or negative documents.

    docs:  {doc_id: (entities, doc_chunk_emb_list)}  — from load_docs()
    Returns: {doc_id: (doc_chunk_emb_list, doc_ent_emb_list)}
      doc_ent_emb_list is [] when no entity overlap (zero vector at load time in dataset.py)
    """
    d = {}
    for doc_id in query_docs:
        if doc_id not in docs or doc_id not in doc_run:
            continue

        is_positive = doc_id in qrels and qrels[doc_id] >= 1
        if is_positive != positive:
            continue

        entities, doc_chunk_emb = docs[doc_id]

        doc_ent_emb = get_entity_centric_doc_embedding(
            doc_entities=entities,
            query_entities=query_entities,
            query_entity_embeddings=query_entity_embeddings,
        )

        if filter_no_entities and doc_ent_emb is None:
            continue

        d[doc_id] = (doc_chunk_emb, doc_ent_emb if doc_ent_emb is not None else [])

    return d


# =============================================================================
#  DATA STRING CREATION
# =============================================================================

def make_data_strings(query, query_id, docs, label, save):
    """Write one JSONL line per (query, doc) pair — POINTWISE."""
    for doc_id, (doc_chunk_embeddings, doc_ent_emb) in docs.items():
        data_line = json.dumps({
            'query':                query,
            'query_id':             query_id,
            'doc_id':               doc_id,
            'doc_chunk_embeddings': doc_chunk_embeddings,
            'doc_ent_emb':          doc_ent_emb,
            'label':                label,
        })
        write_to_file(data_line, save)


def make_pairwise_data_strings(query, query_id, pos_docs, neg_docs, num_negs, save):
    """Write one JSONL line per (pos, neg) pair — PAIRWISE."""
    neg_items   = list(neg_docs.items())
    pairs_added = 0

    for pos_id, (pos_chunk_emb, pos_ent_emb) in pos_docs.items():
        if not neg_items:
            continue
        sampled_negs = random.sample(neg_items, min(num_negs, len(neg_items)))
        for neg_id, (neg_chunk_emb, neg_ent_emb) in sampled_negs:
            data_line = json.dumps({
                'query':                    query,
                'query_id':                 query_id,
                'pos_doc_id':               pos_id,
                'pos_doc_chunk_embeddings': pos_chunk_emb,
                'pos_doc_ent_emb':          pos_ent_emb,
                'neg_doc_id':               neg_id,
                'neg_doc_chunk_embeddings': neg_chunk_emb,
                'neg_doc_ent_emb':          neg_ent_emb,
            })
            write_to_file(data_line, save)
            pairs_added += 1

    return pairs_added


# =============================================================================
#  MAIN DATA CREATION LOOP
# =============================================================================

def create_data(queries, docs, doc_qrels, doc_run, entity_run, entity_embeddings,
                train, balance, save,
                filter_no_entities=False, entity_weighting='minmax',
                train_format='pointwise', negatives_per_pos=1):
    """Build the training or test JSONL file."""
    stats = {
        'total_queries':          0,
        'queries_with_entities':  0,
        'total_examples':         0,
        'examples_with_doc_entities': 0,
        'docs_kept_no_entities':  0,
        'entity_weighting':       entity_weighting,
        'train_format':           train_format if train else 'pointwise (eval)',
    }

    for query_id, query in tqdm(queries.items(), total=len(queries), desc="Queries"):
        stats['total_queries'] += 1

        if query_id not in doc_run or query_id not in entity_run or query_id not in doc_qrels:
            continue

        query_docs = doc_run[query_id]
        qrels      = doc_qrels[query_id]

        query_entities = weight_entity_scores(entity_run[query_id], scheme=entity_weighting)
        query_entity_embeddings = get_query_entity_embeddings(query_entities, entity_embeddings)

        if query_entity_embeddings:
            stats['queries_with_entities'] += 1

        shared_kwargs = dict(
            docs=docs, qrels=qrels,
            query_entities=query_entities,
            query_entity_embeddings=query_entity_embeddings,
            doc_run=query_docs,
            filter_no_entities=filter_no_entities,
        )

        pos_source = set(qrels.keys()) if train else set(query_docs.keys())
        pos_docs   = get_docs(positive=True,  query_docs=pos_source,              **shared_kwargs)
        neg_docs   = get_docs(positive=False, query_docs=set(query_docs.keys()), **shared_kwargs)

        if balance:
            n        = min(len(pos_docs), len(neg_docs))
            pos_docs = dict(list(pos_docs.items())[:n])
            neg_docs = dict(list(neg_docs.items())[:n])

        for _, doc_ent_emb in list(pos_docs.values()) + list(neg_docs.values()):
            if not doc_ent_emb:
                stats['docs_kept_no_entities'] += 1
            else:
                stats['examples_with_doc_entities'] += 1

        if train and train_format == 'pairwise':
            if pos_docs and neg_docs:
                pairs_added = make_pairwise_data_strings(
                    query=query, query_id=query_id,
                    pos_docs=pos_docs, neg_docs=neg_docs,
                    num_negs=negatives_per_pos, save=save,
                )
                stats['total_examples'] += pairs_added
        else:
            make_data_strings(query, query_id, pos_docs, label=1, save=save)
            make_data_strings(query, query_id, neg_docs, label=0, save=save)
            stats['total_examples'] += len(pos_docs) + len(neg_docs)

    return stats


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Make train/test data for DREQ (requires precompute_chunk_embs.py output)."
    )
    parser.add_argument("--queries",     required=True, type=str,
                        help="Queries TSV file (query_id<TAB>query_text).")
    parser.add_argument("--doc-embs",    required=True, type=str,
                        help="Directory produced by precompute_chunk_embs.py "
                             "(contains doc_embs.npy, doc_meta.jsonl.gz, doc_id_to_row.json).")
    parser.add_argument("--qrels",       required=True, type=str,
                        help="Document qrels in TREC format.")
    parser.add_argument("--doc-run",     required=True, type=str,
                        help="Document run file in TREC format.")
    parser.add_argument("--entity-run",  required=True, type=str,
                        help="Entity run file in TREC format.")
    parser.add_argument("--embeddings",  required=True, type=str,
                        help="Wikipedia2Vec entity embeddings (.jsonl.gz).")
    parser.add_argument("--save",        required=True, type=str,
                        help="Output JSONL file.")
    parser.add_argument("--save-stats",  default=None,  type=str,
                        help="Optional path to save statistics JSON.")
    parser.add_argument('--train',       action='store_true',
                        help='Create training data (uses qrels for positives).')
    parser.add_argument('--balance',     action='store_true',
                        help='Balance positive/negative examples.')
    parser.add_argument("--train-format",
                        choices=['pointwise', 'pairwise'], default='pointwise', type=str,
                        help="Training data format. Default: pointwise.")
    parser.add_argument("--negatives-per-pos", default=1, type=int,
                        help="Negatives per positive (pairwise only). Default: 1.")
    parser.add_argument("--filter-no-entities", action='store_true', default=False,
                        help="Filter docs with no entity overlap (old behaviour). "
                             "Default: keep all docs, use zero vector when no overlap.")
    parser.add_argument("--entity-weighting",
                        choices=WEIGHTING_SCHEMES, default='minmax', type=str,
                        help="Entity score weighting scheme. Default: minmax.")
    parser.add_argument("--random-seed", default=42, type=int,
                        help="Random seed. Default: 42.")

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("=" * 60)
    print("DREQ DATA CREATION")
    print("=" * 60)
    print(f"Mode            : {'TRAIN' if args.train else 'TEST/DEV'}")
    if args.train:
        print(f"Format          : {args.train_format.upper()}")
        if args.train_format == 'pairwise':
            print(f"Negs/positive   : {args.negatives_per_pos}")
    print(f"Balance         : {args.balance}")
    print(f"Entity weighting: {args.entity_weighting}")
    if args.entity_weighting == 'raw':
        print("Warning: raw scores — only correct if scores are already in [0,1].")
    if args.filter_no_entities:
        print("Warning: filtering docs with no entity overlap (old behaviour).")
    else:
        print("Keeping all docs — zero entity features when no overlap (recommended).")
    print()

    print("Loading queries...")
    queries = load_queries(args.queries)
    print(f"  {len(queries):,} queries loaded.")

    print("Loading precomputed document embeddings...")
    docs = load_docs(args.doc_embs)

    print("Loading qrels...")
    qrels = read_qrels(args.qrels)

    print("Loading document run...")
    doc_run = read_run(args.doc_run)

    print("Loading entity run...")
    entity_run = read_run(args.entity_run)

    print("Loading entity embeddings...")
    embeddings = load_embeddings(args.embeddings)

    # Clear output file
    with open(args.save, 'w') as f:
        pass

    print("\nCreating data...")
    stats = create_data(
        queries=queries,
        docs=docs,
        doc_qrels=qrels,
        doc_run=doc_run,
        entity_run=entity_run,
        entity_embeddings=embeddings,
        train=args.train,
        balance=args.balance,
        save=args.save,
        filter_no_entities=args.filter_no_entities,
        entity_weighting=args.entity_weighting,
        train_format=args.train_format,
        negatives_per_pos=args.negatives_per_pos,
    )

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total queries processed  : {stats['total_queries']:,}")
    print(f"Queries with entities    : {stats['queries_with_entities']:,}")
    print(f"Total examples created   : {stats['total_examples']:,} "
          f"({'Pairs' if args.train and args.train_format == 'pairwise' else 'Docs'})")
    print(f"Examples with doc entities: {stats['examples_with_doc_entities']:,}")
    print(f"Docs kept (no entity ovlp): {stats['docs_kept_no_entities']:,}")
    print("=" * 60)

    if args.save_stats:
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {args.save_stats}")


if __name__ == '__main__':
    main()