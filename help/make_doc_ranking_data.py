import json
import sys
import math
import numpy as np
import argparse
import collections
from tqdm import tqdm
import gzip
import random
from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, DistilBertModel, T5EncoderModel


# =============================================================================
#  ENCODER
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, pretrained: str) -> None:
        super(Encoder, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        if isinstance(self.encoder, DistilBertModel):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return output[0][:, 0, :]
        elif isinstance(self.encoder, T5EncoderModel):
            last_hidden_state = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return torch.mean(last_hidden_state, dim=1)
        else:
            output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return output[0][:, 0, :]


# =============================================================================
#  FILE I/O HELPERS
# =============================================================================

def read_json_file(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def load_docs(doc_file):
    docs = {}
    with open(doc_file, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            d = json.loads(line)
            if 'chunks' not in d:
                raise ValueError(
                    f"Document '{d['doc_id']}' has no 'chunks' field. "
                    "Run preprocess_docs.py first to pre-chunk the document collection."
                )
            docs[d['doc_id']] = (d['entities'], d['text'], d['chunks'])
    return docs


def read_qrels(qrels_file: str):
    qrels = collections.defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            query_id, _, object_id, relevance = line.strip().split()
            assert object_id not in qrels[query_id]
            qrels[query_id][object_id] = int(relevance)
    return qrels


def read_run(run_file):
    run = collections.defaultdict(dict)
    with open(run_file, 'r') as f:
        for line in f:
            query_id = line.strip().split()[0]
            object_id = line.strip().split()[2]
            score = line.strip().split()[4]
            if object_id not in run[query_id]:
                run[query_id][object_id] = float(score)
    return run


def load_embeddings(embedding_file):
    emb = {}
    print("Loading embeddings...")
    with gzip.open(embedding_file, 'r') as f:
        for line in tqdm(f, total=13032425, desc="Loading embeddings"):
            d = json.loads(line)
            emb[d['entity_id']] = d['embedding']
    print(f"Loaded {len(emb)} entity embeddings")
    return emb


def load_queries(queries_file):
    with open(queries_file, 'r') as f:
        return dict({
            (line.strip().split('\t')[0], line.strip().split('\t')[1])
            for line in f
        })


def write_to_file(data_line, save):
    with open(save, 'a') as f:
        f.write("%s\n" % data_line)


# =============================================================================
#  ENTITY SCORE WEIGHTING
# =============================================================================

WEIGHTING_SCHEMES = ['raw', 'minmax', 'reciprocal', 'uniform', 'log_reciprocal']


def weight_entity_scores(entity_scores, scheme):
    """Re-weight a single query's entity scores according to the chosen scheme."""
    if not entity_scores:
        return entity_scores

    if scheme == 'raw':
        scores = list(entity_scores.values())
        if any(s < 0 for s in scores):
            print("⚠️  Warning: raw weighting selected but negative scores detected. "
                  "Negative weights will flip embedding directions. Consider using 'minmax' instead.")
        return dict(sorted(entity_scores.items(), key=lambda x: x[1], reverse=True))

    if scheme == 'uniform':
        return {eid: 1.0 for eid in entity_scores}

    if scheme in ('reciprocal', 'log_reciprocal'):
        ranked = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        weighted = {}
        for rank, (eid, _) in enumerate(ranked, start=1):
            if scheme == 'reciprocal':
                weighted[eid] = 1.0 / rank
            else:  # log_reciprocal
                weighted[eid] = 1.0 / math.log2(rank + 1)
        return weighted

    if scheme == 'minmax':
        scores = list(entity_scores.values())
        min_s = min(scores)
        max_s = max(scores)
        if max_s > min_s:
            normalized = {eid: (s - min_s) / (max_s - min_s)
                          for eid, s in entity_scores.items()}
        else:
            normalized = {eid: 1.0 for eid in entity_scores}
        return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))

    raise ValueError(f"Unknown weighting scheme: '{scheme}'. Choose from: {WEIGHTING_SCHEMES}")


# =============================================================================
#  DOCUMENT SCORE NORMALIZATION
# =============================================================================

def normalize_doc_scores_per_query(
    doc_scores: Dict[str, float],
    norm: str = "none",
    eps: float = 1e-8,
    warn_on_fallback: bool = True,
    query_id: str = None,
) -> Dict[str, float]:
    """
    Normalize document scores within a query.

    Args:
        doc_scores: {doc_id: score}
        norm: one of {'none', 'zscore', 'rank', 'log1p_zscore'}
        eps: small constant for numerical stability
        warn_on_fallback: if True, prints a warning when log1p_zscore falls back
        query_id: optional query id for logging context

    Returns:
        {doc_id: normalized_score}

    Notes:
        - 'log1p_zscore' is recommended for nonnegative, skewed scores like BM25:
              zscore(log1p(score))
        - If any score is negative, 'log1p_zscore' falls back to plain z-score
          for that query (safe handling for non-BM25 score sources).
    """
    if norm == "none" or not doc_scores:
        return dict(doc_scores)

    items = list(doc_scores.items())
    values = [float(s) for _, s in items]

    if norm == "rank":
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        n = len(sorted_items)
        if n == 1:
            return {sorted_items[0][0]: 1.0}
        return {
            doc_id: 1.0 - (rank / (n - 1))
            for rank, (doc_id, _) in enumerate(sorted_items)
        }

    if norm == "zscore":
        transformed = values
    elif norm == "log1p_zscore":
        has_negative = any(v < 0 for v in values)
        if has_negative:
            if warn_on_fallback:
                qmsg = f" for qid={query_id}" if query_id is not None else ""
                print(
                    f"[make_doc_ranking_data] Warning: negative scores detected{qmsg}; "
                    "falling back from log1p_zscore to zscore for this query."
                )
            transformed = values
        else:
            transformed = [math.log1p(v) for v in values]
    else:
        raise ValueError(
            f"Unknown doc score normalization: {norm}. "
            "Expected one of {'none','zscore','rank','log1p_zscore'}"
        )

    mean_val = sum(transformed) / len(transformed)
    if len(transformed) == 1:
        std_val = 0.0
    else:
        var = sum((x - mean_val) ** 2 for x in transformed) / len(transformed)
        std_val = math.sqrt(var)

    if std_val < eps:
        return {doc_id: 0.0 for doc_id, _ in items}

    return {
        doc_id: (x - mean_val) / (std_val + eps)
        for (doc_id, _), x in zip(items, transformed)
    }


# =============================================================================
#  EMBEDDING HELPERS
# =============================================================================

def get_query_entity_embeddings(query_entities, entity_embeddings):
    """Subset the global embedding table to only the entities in this query."""
    emb = {}
    for entity_id in query_entities:
        if entity_id in entity_embeddings:
            emb[entity_id] = entity_embeddings[entity_id]
    return emb


def get_entity_centric_doc_embedding(doc_entities, query_entities, query_entity_embeddings):
    """
    DREQ-style: compute a single summed entity embedding for the document.
    Entity weights come from the pre-weighted query_entities dict (already
    processed by weight_entity_scores), so no separate weight_method arg needed.
    """
    embeddings = []
    for entity_id in doc_entities:
        entity_id = str(entity_id)
        if entity_id in query_entity_embeddings and entity_id in query_entities:
            entity_embedding = query_entity_embeddings[entity_id]
            if len(entity_embedding) >= 300:
                entity_embedding = entity_embedding[:300]
                entity_weight = query_entities[entity_id]
                embeddings.append(entity_weight * np.array(entity_embedding))

    if not embeddings:
        return None

    # DREQ sums the weighted embeddings into a single vector
    return np.sum(embeddings, axis=0, dtype=np.float32).tolist()


def create_input(text, tokenizer, max_len):
    encoded_dict = tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids']


def get_doc_chunk_embeddings(doc_chunks, encoder, tokenizer, max_len, device):
    embeddings = []
    for chunk in doc_chunks:
        input_ids, attention_mask, token_type_ids = create_input(chunk, tokenizer, max_len)
        with torch.no_grad():
            emb = encoder(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device)
            )
        embeddings.append(emb.squeeze().detach().cpu().tolist())
    return np.mean(np.array(embeddings), axis=0).tolist()


# =============================================================================
#  DOCUMENT COLLECTION
# =============================================================================

def get_docs(docs, qrels, query_entities, query_entity_embeddings,
             positive, query_docs, doc_scores,
             encoder, tokenizer, max_len, device,
             filter_no_entities=False):
    """Collect positive or negative documents with their DREQ embeddings."""
    d = {}

    for doc_id in query_docs:
        if doc_id in docs and doc_id in doc_scores:
            is_positive = doc_id in qrels and qrels[doc_id] >= 1

            if is_positive == positive:
                # DREQ: summed entity embedding (single vector or None)
                doc_ent_emb = get_entity_centric_doc_embedding(
                    doc_entities=docs[doc_id][0],
                    query_entities=query_entities,
                    query_entity_embeddings=query_entity_embeddings,
                )

                if filter_no_entities and doc_ent_emb is None:
                    continue

                # Use pre-computed chunks from preprocess_docs.py
                doc_chunks = docs[doc_id][2]
                doc_chunk_embeddings = get_doc_chunk_embeddings(
                    doc_chunks, encoder, tokenizer, max_len, device
                )

                doc_score = doc_scores[doc_id]
                # Store None as empty list for consistent JSON serialisation
                d[doc_id] = (doc_chunk_embeddings, doc_score, doc_ent_emb if doc_ent_emb is not None else [])

    return d


# =============================================================================
#  DATA STRING CREATION
# =============================================================================

def make_data_strings(query, query_id, docs, label, save):
    """Serialise examples to JSONL, one line per (query, doc) pair (POINTWISE)."""
    for doc_id, (doc_chunk_embeddings, doc_score, doc_ent_emb) in docs.items():
        data_line = json.dumps({
            'query': query,
            'query_id': query_id,
            'doc_id': doc_id,
            'doc_chunk_embeddings': doc_chunk_embeddings,
            'doc_score': doc_score,
            'doc_ent_emb': doc_ent_emb,
            'label': label
        })
        write_to_file(data_line, save)


def make_pairwise_data_strings(query, query_id, pos_docs, neg_docs, num_negs, save):
    """Serialise examples to JSONL in pairwise triplet format (PAIRWISE)."""
    neg_items = list(neg_docs.items())
    pairs_added = 0

    for pos_id, (pos_chunk_emb, pos_score, pos_ent_emb) in pos_docs.items():
        if not neg_items:
            continue

        sampled_negs = random.sample(neg_items, min(num_negs, len(neg_items)))

        for neg_id, (neg_chunk_emb, neg_score, neg_ent_emb) in sampled_negs:
            data_line = json.dumps({
                'query': query,
                'query_id': query_id,

                # Positive document features
                'pos_doc_id': pos_id,
                'pos_doc_chunk_embeddings': pos_chunk_emb,
                'pos_doc_score': pos_score,
                'pos_doc_ent_emb': pos_ent_emb,

                # Negative document features
                'neg_doc_id': neg_id,
                'neg_doc_chunk_embeddings': neg_chunk_emb,
                'neg_doc_score': neg_score,
                'neg_doc_ent_emb': neg_ent_emb,
            })
            write_to_file(data_line, save)
            pairs_added += 1

    return pairs_added


# =============================================================================
#  MAIN DATA CREATION LOOP
# =============================================================================

def create_data(queries, docs, doc_qrels, doc_run, entity_run, entity_embeddings,
                train, balance, save, encoder, tokenizer, max_len, device,
                filter_no_entities=False, entity_weighting='minmax',
                train_format='pointwise', negatives_per_pos=1, doc_score_norm='none'):
    """Build the training or test JSONL file."""
    stats = {
        'total_queries': 0,
        'queries_with_entities': 0,
        'total_examples': 0,
        'examples_with_doc_entities': 0,
        'docs_filtered_no_entities': 0,
        'docs_kept_no_entities': 0,
        'entity_weighting': entity_weighting,
        'train_format': train_format if train else 'pointwise (eval is always pointwise)',
        'doc_score_norm': doc_score_norm,
    }

    for query_id, query in tqdm(queries.items(), total=len(queries)):
        stats['total_queries'] += 1

        if query_id not in doc_run or query_id not in entity_run or query_id not in doc_qrels:
            continue

        query_docs = doc_run[query_id]
        qrels = doc_qrels[query_id]

        # Per-query normalization of document scores
        query_docs = normalize_doc_scores_per_query(
            query_docs,
            norm=doc_score_norm,
            warn_on_fallback=True,
            query_id=query_id
        )

        query_entities = weight_entity_scores(entity_run[query_id], scheme=entity_weighting)

        query_entity_embeddings = get_query_entity_embeddings(
            query_entities=query_entities,
            entity_embeddings=entity_embeddings
        )

        if query_entity_embeddings:
            stats['queries_with_entities'] += 1

        shared_kwargs = dict(
            docs=docs, qrels=qrels,
            query_entities=query_entities,
            query_entity_embeddings=query_entity_embeddings,
            doc_scores=query_docs,
            encoder=encoder,
            tokenizer=tokenizer, max_len=max_len, device=device,
            filter_no_entities=filter_no_entities,
        )

        # Positive documents
        pos_source = set(qrels.keys()) if train else set(query_docs.keys())
        pos_docs = get_docs(positive=True, query_docs=pos_source, **shared_kwargs)

        # Negative documents (always from the run file)
        neg_docs = get_docs(positive=False, query_docs=set(query_docs.keys()), **shared_kwargs)

        if balance:
            n = min(len(pos_docs), len(neg_docs))
            pos_docs = dict(list(pos_docs.items())[:n])
            neg_docs = dict(list(neg_docs.items())[:n])

        # Accumulate stats on entity overlap
        for _, doc_score, doc_ent_emb in list(pos_docs.values()) + list(neg_docs.values()):
            if not doc_ent_emb:
                stats['docs_kept_no_entities'] += 1
            else:
                stats['examples_with_doc_entities'] += 1

        # Write out data (pairwise or pointwise)
        if train and train_format == 'pairwise':
            if pos_docs and neg_docs:
                pairs_added = make_pairwise_data_strings(
                    query=query,
                    query_id=query_id,
                    pos_docs=pos_docs,
                    neg_docs=neg_docs,
                    num_negs=negatives_per_pos,
                    save=save
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
    parser = argparse.ArgumentParser("Make train/test data for DREQ.")
    parser.add_argument("--queries", help="Queries file.", required=True, type=str)
    parser.add_argument("--docs", help="Document file.", required=True, type=str)
    parser.add_argument("--qrels", help="Document qrels.", required=True, type=str)
    parser.add_argument("--doc-run", help="Document run file.", required=True, type=str)
    parser.add_argument("--entity-run", help="Entity run file.", required=True, type=str)
    parser.add_argument("--embeddings", help="Wikipedia2Vec entity embeddings file.", required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--encoder', help='Name of model (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). '
                                          'Default: bert.', type=str, default='bert')
    parser.add_argument('--train', help='Is this train data? Default: False.', action='store_true')
    parser.add_argument('--balance', help='Should the data be balanced? Default: False.', action='store_true')
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--save-stats", help="Save statistics JSON file.", type=str)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')

    # Pairwise generation
    parser.add_argument("--train-format",
                        help="Format for training data: 'pointwise' or 'pairwise'. Default: pointwise",
                        choices=['pointwise', 'pairwise'], default='pointwise', type=str)
    parser.add_argument("--negatives-per-pos",
                        help="How many negative docs to sample per positive doc (pairwise only). Default: 1",
                        default=1, type=int)

    # Entity / doc filtering and weighting
    parser.add_argument("--filter-no-entities",
                        help="Filter out documents with no entity overlap (old behaviour).",
                        action='store_true', default=False)
    parser.add_argument("--entity-weighting",
                        help="How to weight entity scores before scaling embeddings.",
                        choices=WEIGHTING_SCHEMES, default='minmax', type=str)
    parser.add_argument("--doc-score-norm",
                        type=str, default="none",
                        choices=["none", "zscore", "rank", "log1p_zscore"],
                        help=(
                            "Per-query normalization for document scores. "
                            "'log1p_zscore' (recommended for BM25) applies zscore(log1p(score)) "
                            "for nonneg scores and falls back to zscore if negative scores detected."
                        ))
    parser.add_argument("--random-seed", help="Random seed for reproducibility. Default: 42", type=int, default=42)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("=" * 60)
    print("DREQ DATA CREATION")
    print("=" * 60)

    print(f"{'✅ Creating train data...' if args.train else '✅ Creating test data...'}")
    print(f"[make_doc_ranking_data] doc_score_norm = {args.doc_score_norm}")

    if args.train:
        print(f"✅ Training data format: {args.train_format.upper()}")
        if args.train_format == 'pairwise':
            print(f"✅ Generating {args.negatives_per_pos} negatives per positive document.")

    print(f"{'✅ Dataset will be balanced.' if args.balance else '✅ Dataset will be unbalanced.'}")
    print(f"✅ Entity score weighting: {args.entity_weighting}")
    print(f"✅ Document score normalization: {args.doc_score_norm}")

    if args.entity_weighting == 'raw':
        print("⚠️  Using raw scores — only correct if scores are already in [0,1].")

    if args.filter_no_entities:
        print("⚠️  Filtering docs with no entity overlap (old behaviour).")
    else:
        print("✅ Keeping all docs; zero entity features used when there is no overlap (recommended).")

    model_map = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base',
        'ernie': 'nghuyong/ernie-2.0-base-en',
        'electra': 'google/electra-small-discriminator',
        'conv-bert': 'YituTech/conv-bert-base',
        't5': 't5-base'
    }
    cuda_device = 'cuda:' + str(args.cuda)
    print(f'CUDA Device: {cuda_device}')
    device = torch.device(cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {device}')
    print(f'Model ==> {args.encoder}')
    pretrain = model_map[args.encoder]
    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)
    encoder = Encoder(pretrained=pretrain)
    encoder.to(device)
    encoder.eval()

    print("Loading queries...")
    queries = load_queries(args.queries)
    print("Loading documents...")
    docs = load_docs(args.docs)
    print("Loading qrels...")
    qrels = read_qrels(args.qrels)
    print("Loading document run...")
    doc_run = read_run(args.doc_run)
    print("Loading entity run...")
    entity_run = read_run(args.entity_run)
    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings)

    # Clear output file before appending
    with open(args.save, 'w') as f:
        pass

    print("Creating data...")
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
        encoder=encoder,
        tokenizer=tokenizer,
        max_len=args.max_len,
        device=device,
        filter_no_entities=args.filter_no_entities,
        entity_weighting=args.entity_weighting,
        train_format=args.train_format,
        negatives_per_pos=args.negatives_per_pos,
        doc_score_norm=args.doc_score_norm,
    )
    print("[Done].")

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total queries processed:       {stats['total_queries']}")
    print(f"Queries with entities:         {stats['queries_with_entities']}")
    print(f"Total examples created:        {stats['total_examples']} "
          f"({'Pairs' if args.train and args.train_format == 'pairwise' else 'Docs'})")
    print(f"Examples with doc entities:    {stats['examples_with_doc_entities']}")
    print(f"Docs kept (no entity overlap): {stats['docs_kept_no_entities']}")

    if args.save_stats:
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.save_stats}")


if __name__ == '__main__':
    main()