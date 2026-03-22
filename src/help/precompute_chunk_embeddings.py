"""
precompute_chunk_embs.py
========================

Chunk a document corpus with SpacyPassageChunker, encode each chunk with any
HF encoder or sentence-transformers model, mean-pool chunks per document into
a single [D] vector, and save the result in a numpy-based layout for fast loading.

Replaces both preprocess_docs.py AND the encoder step inside
make_doc_ranking_data_dreq.py. Generic enough to be reused for any IR corpus.

Output layout (--out-dir):
    doc_embs.npy          float16/float32  [N_docs, D]  — one row per doc
    doc_meta.jsonl.gz     one JSON per doc: {doc_id, entities, row_idx}
    doc_id_to_row.json    {doc_id: row_idx}  for O(1) lookup
    manifest.json         config fingerprint for compatibility checking

JSONL field mapping
-------------------
By default expects: doc_id, entities, text.
Use --id-key / --text-key to map to different field names.
Use --title-key to prepend a title field to each document.

Encoder quick reference
-----------------------
BERT / RoBERTa / BGE / E5-base (encoder-only):
  --encoder bert  (or --pretrained-model BAAI/bge-base-en-v1.5)
  --pooling cls   (DREQ default) or --pooling mean

Decoder-based embedding models (GTE-Qwen2, Qwen3-Embedding, NV-Embed):
  --pretrained-model Alibaba-NLP/gte-Qwen2-7B-instruct
  --pooling last_token --normalize --trust-remote-code

Mistral-family (e5-mistral-7b-instruct, SFR-Embedding-Mistral):
  --pretrained-model intfloat/e5-mistral-7b-instruct
  --pooling last_token --normalize --append-eos

SBERT models:
  --encoder-type sbert --pretrained-model sentence-transformers/all-mpnet-base-v2
  (--pooling is ignored for sbert, a warning is emitted)

Usage (DREQ default):
    python precompute_chunk_embs.py \
        --docs         /path/to/docs.jsonl \
        --out-dir      /path/to/precomputed/bert_spacy10s5 \
        --encoder      bert \
        --max-len      512 \
        --max-sent-len 10 \
        --stride       5 \
        --dtype        float16 \
        --use-cuda

Usage (arbitrary HF model):
    python precompute_chunk_embs.py \
        --docs             /path/to/docs.jsonl \
        --out-dir          /path/to/precomputed/bge_spacy10s5 \
        --pretrained-model BAAI/bge-base-en-v1.5 \
        --pooling          cls \
        --normalize \
        --doc-prefix       "passage: " \
        --use-cuda
"""

import argparse
import gzip
import hashlib
import inspect
import json
import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel, DistilBertModel, T5EncoderModel
from spacy_passage_chunker import SpacyPassageChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("precompute_chunk_embs")


# =============================================================================
#  MODEL MAP  (alias -> HF name)
# =============================================================================

MODEL_MAP = {
    'bert':       'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta':    'roberta-base',
    'deberta':    'microsoft/deberta-base',
    'ernie':      'nghuyong/ernie-2.0-base-en',
    'electra':    'google/electra-small-discriminator',
    'conv-bert':  'YituTech/conv-bert-base',
    't5':         't5-base',
}


# =============================================================================
#  TOKENIZER HELPERS
# =============================================================================

def _prepare_tokenizer_for_embedding(tokenizer) -> None:
    """
    In-place fixes for tokenizers not set up for embedding (vs. generation):
      1. Decoder-family tokenizers often have pad_token=None → set to eos_token.
      2. Force padding_side='right' — left-padding (generation default for causal
         models) breaks CLS pooling and last_token index calculation.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            log.info(f"[tokenizer] pad_token was None; set to eos_token '{tokenizer.eos_token}'.")
        else:
            raise RuntimeError(
                "Tokenizer has neither pad_token nor eos_token. "
                "Set a pad token manually or use a different model."
            )
    if tokenizer.padding_side != 'right':
        log.info(f"[tokenizer] padding_side was '{tokenizer.padding_side}'; forcing 'right' for embedding.")
        tokenizer.padding_side = 'right'


def _model_accepts_token_type_ids(model) -> bool:
    """
    Check whether model.forward() accepts token_type_ids.
    Works for both our Encoder wrapper and raw AutoModel instances.
    """
    try:
        inner = model.encoder if hasattr(model, 'encoder') else model
        sig = inspect.signature(inner.forward)
        return 'token_type_ids' in sig.parameters
    except Exception:
        inner = model.encoder if hasattr(model, 'encoder') else model
        model_type = getattr(getattr(inner, 'config', None), 'model_type', '')
        no_tti = {'roberta', 'xlm-roberta', 'xlm_roberta', 'distilbert',
                  'mistral', 'llama', 'qwen2', 'gemma', 'gemma2', 'phi', 'falcon'}
        return not any(f in model_type.lower() for f in no_tti)


# =============================================================================
#  HF ENCODER
# =============================================================================

class Encoder(nn.Module):
    """
    Wraps any HF AutoModel and exposes a unified forward() that returns a
    [B, D] embedding tensor using the specified pooling strategy.
    """

    def __init__(self, pretrained: str, trust_remote_code: bool = False,
                 use_sdpa: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(
            self.pretrained, trust_remote_code=trust_remote_code
        )
        load_kwargs = dict(config=self.config, trust_remote_code=trust_remote_code)

        if pretrained.endswith('t5-base') or 't5' in pretrained.lower():
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, **load_kwargs)
        elif use_sdpa:
            try:
                self.encoder = AutoModel.from_pretrained(
                    self.pretrained, attn_implementation='sdpa', **load_kwargs
                )
                log.info("[encoder] Loaded with SDPA attention.")
            except (ValueError, TypeError):
                log.warning("[encoder] SDPA not supported; falling back to eager attention.")
                self.encoder = AutoModel.from_pretrained(self.pretrained, **load_kwargs)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, **load_kwargs)

    def forward(self, input_ids, attention_mask, token_type_ids,
                pooling: str = 'cls') -> torch.Tensor:
        from transformers import T5EncoderModel

        if isinstance(self.encoder, DistilBertModel):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state                                   # [B, T, D]
        elif isinstance(self.encoder, T5EncoderModel):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state                                   # [B, T, D]
        else:
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kwargs['token_type_ids'] = token_type_ids
            hidden = self.encoder(**kwargs).last_hidden_state     # [B, T, D]

        hidden = hidden.float()

        if pooling == 'cls':
            return hidden[:, 0, :]                                # [B, D]

        elif pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()           # [B, T, 1]
            denom = mask.sum(dim=1).clamp(min=1e-9)               # [B, 1]
            return (hidden * mask).sum(dim=1) / denom             # [B, D]

        elif pooling == 'last_token':
            # padding_side is forced to 'right', so last real token is at
            # index (sum of mask - 1).
            lengths  = attention_mask.sum(dim=1)                  # [B]
            last_idx = (lengths - 1).clamp(min=0).long()         # [B]
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, last_idx, :]                 # [B, D]

        else:
            raise ValueError(f"Unknown pooling: '{pooling}'. Choose cls / mean / last_token.")


# =============================================================================
#  TOKENISATION
# =============================================================================

def _tokenize_with_optional_eos(texts, tokenizer, max_len: int, append_eos: bool) -> dict:
    """
    Tokenize a batch of texts, optionally appending EOS before padding.
    Required for Mistral-family models (e5-mistral, SFR-Embedding-Mistral) whose
    tokenizers do NOT auto-add EOS. GTE-Qwen2/Qwen3/BERT do not need this.
    """
    if not append_eos:
        return tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_len, return_tensors='pt',
        )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(
            "--append-eos was set but tokenizer has no eos_token_id. "
            "Only use this for Mistral-family models."
        )

    enc_no_pad = tokenizer(
        texts, padding=False, truncation=True,
        max_length=max_len - 1,
        return_attention_mask=False, return_tensors=None,
    )
    enc_no_pad['input_ids'] = [ids + [eos_id] for ids in enc_no_pad['input_ids']]
    return tokenizer.pad(enc_no_pad, padding=True,
                         return_attention_mask=True, return_tensors='pt')


def create_input_single(text: str, tokenizer, max_len: int,
                        accepts_tti: bool, append_eos: bool,
                        doc_prefix: str = '') -> tuple:
    """Tokenize a single chunk for one-at-a-time encoding (original DREQ style)."""
    full_text = f"{doc_prefix}{text}" if doc_prefix else text
    enc = _tokenize_with_optional_eos([full_text], tokenizer, max_len, append_eos)
    input_ids      = enc['input_ids']
    attention_mask = enc['attention_mask']
    token_type_ids = enc.get('token_type_ids')
    if not accepts_tti or token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    return input_ids, attention_mask, token_type_ids


# =============================================================================
#  SBERT ENCODING
# =============================================================================

def encode_chunks_sbert(chunks, model_name: str, device: str,
                        normalize: bool, use_amp: bool,
                        use_compile: bool, doc_prefix: str) -> np.ndarray:
    """Encode a list of chunks with sentence-transformers and mean-pool."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    load_kwargs = {}
    if use_amp and device.startswith('cuda'):
        load_kwargs['model_kwargs'] = {'torch_dtype': torch.bfloat16}

    model = SentenceTransformer(model_name, device=device, **load_kwargs)

    if use_compile and device.startswith('cuda'):
        try:
            model[0].auto_model = torch.compile(model[0].auto_model)
        except Exception as e:
            log.warning(f"[sbert] torch.compile() failed ({e}); using eager.")

    texts = [f"{doc_prefix}{c}" if doc_prefix else c for c in chunks]
    embs = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=normalize,
    ).astype(np.float32)                                           # [n_chunks, D]
    return np.mean(embs, axis=0)                                   # [D]


# =============================================================================
#  CHUNK ENCODING  (HF path — spaCy chunks → mean-pooled doc embedding)
# =============================================================================

def encode_chunks_hf(chunks, encoder: Encoder, tokenizer, max_len: int,
                     device: torch.device, accepts_tti: bool,
                     pooling: str, normalize: bool, append_eos: bool,
                     use_amp: bool, amp_dtype: torch.dtype,
                     doc_prefix: str) -> np.ndarray:
    """
    Encode a list of text chunks one-at-a-time and mean-pool into a [D] vector.
    Keeps the original DREQ encoding loop style.
    """
    embeddings = []
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    for chunk in chunks:
        input_ids, attention_mask, token_type_ids = create_input_single(
            chunk, tokenizer, max_len, accepts_tti, append_eos, doc_prefix
        )
        tti_arg = token_type_ids.to(device) if accepts_tti else None
        with torch.inference_mode(), amp_ctx:
            emb = encoder(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=tti_arg,
                pooling=pooling,
            )                                                      # [1, D]
        vec = emb.squeeze(0).float().cpu().numpy()
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        embeddings.append(vec)

    return np.mean(np.stack(embeddings), axis=0)                  # [D]


# =============================================================================
#  FILE I/O
# =============================================================================

def iter_docs(doc_file: str, id_key: str, text_key: str,
              title_key: str = None, max_text_chars: int = None):
    """Stream docs from a JSONL file with configurable field mapping."""
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            raw_id = obj.get(id_key)
            if raw_id is None:
                continue
            doc_id = str(raw_id)

            text = str(obj.get(text_key, '') or '').replace('\n', ' ')
            if max_text_chars and max_text_chars > 0:
                text = text[:max_text_chars]

            title = ''
            if title_key:
                title = str(obj.get(title_key, '') or '')

            # Always carry entities through if present (DREQ-specific field)
            entities = obj.get('entities', [])

            yield doc_id, text, title, entities


def count_docs(doc_file: str) -> int:
    n = 0
    with open(doc_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# =============================================================================
#  MANIFEST
# =============================================================================

def build_manifest(args, pretrain: str, dim: int) -> dict:
    cfg = {
        'docs':           args.docs,
        'id_key':         args.id_key,
        'text_key':       args.text_key,
        'title_key':      args.title_key,
        'max_text_chars': args.max_text_chars,
        'pretrained':     pretrain,
        'encoder_type':   args.encoder_type,
        'pooling':        args.pooling if args.encoder_type == 'hf' else 'sbert_internal',
        'normalize':      args.normalize,
        'append_eos':     args.append_eos if args.encoder_type == 'hf' else False,
        'doc_prefix':     args.doc_prefix or '',
        'max_len':        args.max_len if args.encoder_type == 'hf' else None,
        'max_sent_len':   args.max_sent_len,
        'stride':         args.stride,
        'prepend_title':  args.prepend_title,
        'dtype':          args.dtype,
        'dim':            dim,
    }
    fingerprint = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode('utf-8')
    ).hexdigest()
    cfg['fingerprint_sha256'] = fingerprint
    return cfg


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precompute spaCy-chunked document embeddings for IR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("--docs",    required=True,
                        help="Input docs JSONL file.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory (created if it does not exist).")

    # ── JSONL field mapping ───────────────────────────────────────────────────
    parser.add_argument("--id-key",   default="doc_id",
                        help="JSON key for document ID. Default: doc_id.")
    parser.add_argument("--text-key", default="text",
                        help="JSON key for document text. Default: text.")
    parser.add_argument("--title-key", default=None,
                        help="Optional JSON key for title/name. "
                             "If set alongside --prepend-title, prepended to text before chunking.")
    parser.add_argument("--max-text-chars", type=int, default=None,
                        help="Truncate document text to this many characters before chunking.")

    # ── Encoder ───────────────────────────────────────────────────────────────
    parser.add_argument("--encoder-type", choices=['hf', 'sbert'], default='hf',
                        help="Encoder backend. 'hf' uses HF AutoModel; "
                             "'sbert' uses sentence-transformers. Default: hf.")
    parser.add_argument("--encoder",
                        choices=list(MODEL_MAP.keys()), default='bert',
                        help=f"Encoder alias ({list(MODEL_MAP.keys())}). "
                             "Default: bert. Ignored if --pretrained-model is set.")
    parser.add_argument("--pretrained-model", default=None,
                        help="Override --encoder with any HF/SBERT model name or local path, "
                             "e.g. BAAI/bge-base-en-v1.5.")
    parser.add_argument("--pooling",
                        choices=['cls', 'mean', 'last_token'], default='cls',
                        help="Per-chunk token pooling strategy (HF only; ignored for sbert). "
                             "Use 'last_token' for decoder-based embedding models "
                             "(GTE-Qwen2, e5-mistral, etc.). Default: cls.")
    parser.add_argument("--normalize", action='store_true',
                        help="L2-normalise each chunk embedding before mean-pooling across chunks.")
    parser.add_argument("--append-eos", action='store_true',
                        help="Append EOS token before padding (HF only). "
                             "Required for Mistral-family models (e5-mistral, SFR-Embedding-Mistral) "
                             "whose tokenizers do NOT auto-add EOS. "
                             "Do NOT use for BERT, RoBERTa, GTE-Qwen2, or Qwen3.")
    parser.add_argument("--trust-remote-code", action='store_true',
                        help="Pass trust_remote_code=True to from_pretrained. "
                             "Required for GTE-Qwen2, NV-Embed-v2, and other custom-code models.")
    parser.add_argument("--doc-prefix", default='',
                        help="Optional prefix prepended to every chunk before encoding. "
                             "E.g. 'passage: ' for E5-base/large. Default: none.")

    # ── Chunking (SpacyPassageChunker) ────────────────────────────────────────
    parser.add_argument('--max-sent-len', default=10, type=int,
                        help='Max sentences per passage chunk. Default: 10.')
    parser.add_argument('--stride', default=5, type=int,
                        help='Sentence stride between chunks. Default: 5.')
    parser.add_argument("--prepend-title", action='store_true',
                        help="Prepend --title-key value to document text before chunking.")

    # ── Tokeniser / encoding (HF only) ───────────────────────────────────────
    parser.add_argument('--max-len', default=512, type=int,
                        help='Max token length for HF encoder. Default: 512.')

    # ── Storage ───────────────────────────────────────────────────────────────
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
                        help='Embedding storage dtype. float16 halves disk/memory. Default: float16.')
    parser.add_argument("--flush-every", type=int, default=10000,
                        help="Flush memmap to disk every N docs. Default: 10000.")

    # ── Device / performance ─────────────────────────────────────────────────
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device number. Default: 0.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA if available.')
    parser.add_argument('--dtype-amp', choices=['bf16', 'fp16', 'fp32'], default='bf16',
                        help='AMP dtype for HF encoding. bf16 recommended for Ampere. Default: bf16.')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP entirely.')
    parser.add_argument('--no-tf32', action='store_true',
                        help='Disable TF32 on Ampere GPUs.')
    parser.add_argument('--no-sdpa', action='store_true',
                        help='Disable SDPA (scaled dot-product attention). '
                             'Use for models that do not support attn_implementation=sdpa.')
    parser.add_argument('--use-compile', action='store_true',
                        help='Enable torch.compile() on the encoder for extra speed on Ampere. '
                             'Adds a one-time warm-up cost. Not supported on all models.')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.encoder_type == 'sbert' and args.pooling != 'cls':
        log.warning(f"[sbert] --pooling={args.pooling} is ignored; "
                    "sentence-transformers handles pooling internally.")
    if args.append_eos and args.encoder_type == 'sbert':
        log.warning("[sbert] --append-eos is ignored for --encoder-type sbert.")

    # ── Device + Ampere optimisations ────────────────────────────────────────
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )
    if device.type == 'cuda' and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("TF32 enabled.")

    amp_dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = amp_dtype_map[args.dtype_amp]
    use_amp   = (device.type == 'cuda') and not args.no_amp

    # ── Resolve pretrained model name ────────────────────────────────────────
    pretrain = args.pretrained_model if args.pretrained_model else MODEL_MAP.get(args.encoder, args.encoder)
    log.info(f"Encoder: {pretrain}  |  type: {args.encoder_type}")
    log.info(f"Device:  {device}  |  AMP: {use_amp} ({args.dtype_amp})")

    # ── Output paths ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    embs_path     = os.path.join(args.out_dir, 'doc_embs.npy')
    mmap_path     = os.path.join(args.out_dir, 'doc_embs.mmap')
    meta_path     = os.path.join(args.out_dir, 'doc_meta.jsonl.gz')
    id2row_path   = os.path.join(args.out_dir, 'doc_id_to_row.json')
    manifest_path = os.path.join(args.out_dir, 'manifest.json')

    # ── Load encoder (HF path) ───────────────────────────────────────────────
    encoder = tokenizer = accepts_tti = None
    if args.encoder_type == 'hf':
        log.info("Loading tokenizer and encoder (once)...")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrain, model_max_length=args.max_len,
            trust_remote_code=args.trust_remote_code,
        )
        _prepare_tokenizer_for_embedding(tokenizer)

        encoder = Encoder(
            pretrained=pretrain,
            trust_remote_code=args.trust_remote_code,
            use_sdpa=not args.no_sdpa,
        )
        encoder.to(device)
        encoder.eval()

        if args.use_compile and device.type == 'cuda':
            try:
                encoder = torch.compile(encoder)
                log.info("[hf] torch.compile() enabled.")
            except Exception as e:
                log.warning(f"[hf] torch.compile() failed ({e}); using eager.")

        accepts_tti = _model_accepts_token_type_ids(encoder)
        if not accepts_tti:
            log.info("[hf] Model does not accept token_type_ids — zero tensor will be used.")

        # Probe embedding dim
        with torch.inference_mode():
            dummy_ids  = torch.zeros(1, 16, dtype=torch.long, device=device)
            dummy_mask = torch.ones(1, 16, dtype=torch.long, device=device)
            dim = encoder(dummy_ids, dummy_mask, None, pooling=args.pooling).shape[-1]
        log.info(f"Embedding dim: {dim}")
    else:
        # SBERT: probe dim by encoding one dummy string
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        _sbert_probe = SentenceTransformer(pretrain, device=str(device))
        dim = _sbert_probe.encode(["dummy"], convert_to_numpy=True).shape[1]
        del _sbert_probe
        log.info(f"[sbert] Embedding dim: {dim}")

    # ── Load spaCy chunker ───────────────────────────────────────────────────
    log.info(f"Loading SpacyPassageChunker "
             f"(max_sent_len={args.max_sent_len}, stride={args.stride})...")
    chunker = SpacyPassageChunker(max_len=args.max_sent_len, stride=args.stride)

    # ── Pass 1: count docs → pre-allocate memmap ─────────────────────────────
    log.info("Pass 1/2: counting documents...")
    total_docs = count_docs(args.docs)
    log.info(f"Total documents: {total_docs:,}")

    store_dtype = np.float16 if args.dtype == 'float16' else np.float32
    log.info(f"Allocating memmap: ({total_docs:,}, {dim})  dtype={args.dtype}")
    embs_mmap = np.memmap(mmap_path, mode='w+', dtype=store_dtype, shape=(total_docs, dim))

    # ── Pass 2: chunk → encode → write ───────────────────────────────────────
    log.info("Pass 2/2: chunking + encoding + writing...")

    doc_id_to_row = {}
    total_chunks  = 0
    no_chunk_docs = 0
    row = 0

    meta_f = gzip.open(meta_path, 'wt', encoding='utf-8')

    for doc_id, text, title, entities in tqdm(
        iter_docs(args.docs, args.id_key, args.text_key, args.title_key, args.max_text_chars),
        total=total_docs, desc="Processing docs"
    ):
        # Optionally prepend title
        if args.prepend_title and title:
            text = f"{title}. {text}".strip()

        # spaCy sentence chunking
        chunker.tokenize_document(text)
        chunks = chunker.chunk_document()

        if not chunks:
            no_chunk_docs += 1
            chunks = [text]   # fallback: full doc as single chunk

        total_chunks += len(chunks)

        # Encode chunks → mean-pooled doc embedding [D]
        if args.encoder_type == 'hf':
            doc_emb = encode_chunks_hf(
                chunks=chunks,
                encoder=encoder,
                tokenizer=tokenizer,
                max_len=args.max_len,
                device=device,
                accepts_tti=accepts_tti,
                pooling=args.pooling,
                normalize=args.normalize,
                append_eos=args.append_eos,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                doc_prefix=args.doc_prefix,
            )
        else:
            doc_emb = encode_chunks_sbert(
                chunks=chunks,
                model_name=pretrain,
                device=str(device),
                normalize=args.normalize,
                use_amp=use_amp,
                use_compile=args.use_compile,
                doc_prefix=args.doc_prefix,
            )

        embs_mmap[row] = doc_emb.astype(store_dtype)
        doc_id_to_row[doc_id] = row
        meta_f.write(json.dumps({
            'doc_id':   doc_id,
            'entities': entities,
            'row_idx':  row,
        }, ensure_ascii=False) + '\n')

        row += 1

        if row % args.flush_every == 0:
            embs_mmap.flush()
            meta_f.flush()
            log.info(f"  Flushed at row {row:,}")

    meta_f.close()

    # ── Save final outputs ───────────────────────────────────────────────────
    log.info("Saving final .npy embeddings...")
    final_embs = np.asarray(embs_mmap[:row])
    np.save(embs_path, final_embs)
    log.info(f"Saved: {embs_path}  shape={final_embs.shape}  dtype={final_embs.dtype}")

    with open(id2row_path, 'w', encoding='utf-8') as f:
        json.dump(doc_id_to_row, f, ensure_ascii=False)
    log.info(f"Saved: {id2row_path}  ({len(doc_id_to_row):,} docs)")

    manifest = build_manifest(args, pretrain=pretrain, dim=dim)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
    log.info(f"Saved: {manifest_path}")
    log.info(f"Fingerprint: {manifest['fingerprint_sha256']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Documents processed  : {row:,}")
    print(f"Docs with no chunks  : {no_chunk_docs:,} (fell back to full text)")
    print(f"Total chunks encoded : {total_chunks:,}")
    print(f"Avg chunks / doc     : {total_chunks / max(row, 1):.2f}")
    print(f"Embedding shape      : ({row:,}, {dim})")
    print(f"Storage dtype        : {args.dtype}")
    print(f"Encoder              : {pretrain}")
    print(f"Pooling              : {args.pooling if args.encoder_type == 'hf' else 'sbert_internal'}")
    print(f"Output dir           : {args.out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()