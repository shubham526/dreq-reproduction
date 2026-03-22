from typing import Dict, Any, List, Tuple
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Entity embedding dimension produced by Wikipedia2Vec
ENTITY_EMB_DIM = 300


class DocRankingDataset(Dataset):
    def __init__(self, dataset: str, tokenizer, train: bool, max_len: int):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._train = train
        self._read_data()
        self._count = len(self._examples)

        # Auto-detect pairwise vs pointwise from the first example
        self.format = 'pairwise' if (
            self._count > 0 and 'pos_doc_chunk_embeddings' in self._examples[0]
        ) else 'pointwise'

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        query_input_ids, query_attn_mask, query_token_type_ids = self._create_input(example['query'])

        if self.format == 'pairwise':
            return {
                'query_input_ids':       query_input_ids,
                'query_attention_mask':  query_attn_mask,
                'query_token_type_ids':  query_token_type_ids,

                'pos_doc_chunk_embeddings': example['pos_doc_chunk_embeddings'],
                'pos_doc_ent_emb':          example['pos_doc_ent_emb'],

                'neg_doc_chunk_embeddings': example['neg_doc_chunk_embeddings'],
                'neg_doc_ent_emb':          example['neg_doc_ent_emb'],
            }

        # Pointwise
        res = {
            'query_input_ids':       query_input_ids,
            'query_attention_mask':  query_attn_mask,
            'query_token_type_ids':  query_token_type_ids,
            'doc_chunk_embeddings':  example['doc_chunk_embeddings'],
            'doc_ent_emb':           example['doc_ent_emb'],
            'label':                 example['label'],
        }
        if not self._train:
            res['query_id'] = example['query_id']
            res['doc_id']   = example['doc_id']
        return res

    def _read_data(self):
        with open(self._dataset, 'r', encoding='utf-8') as f:
            self._examples = [json.loads(line) for line in tqdm(f, desc="Loading dataset")]

    def _create_input(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            max_length=self._max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=None,
        )
        input_ids      = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        # Fallback for tokenizers that omit token_type_ids (e.g. RoBERTa)
        token_type_ids = encoded.get('token_type_ids') or [0] * len(input_ids)
        return input_ids, attention_mask, token_type_ids

    @staticmethod
    def _to_entity_tensor(emb_list) -> torch.Tensor:
        """
        Convert a doc_ent_emb value to a [ENTITY_EMB_DIM] float32 tensor.
        DREQ stores either a [300] list (entity overlap found) or [] (no overlap).
        Empty list -> zero vector so downstream doc_ranking receives a fixed-shape input.
        """
        if len(emb_list) == 0:
            return torch.zeros(ENTITY_EMB_DIM, dtype=torch.float32)
        return torch.tensor(emb_list, dtype=torch.float32)

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Query tokens — always present in both formats
        query_input_ids      = torch.tensor([b['query_input_ids']      for b in batch], dtype=torch.long)
        query_attention_mask = torch.tensor([b['query_attention_mask']  for b in batch], dtype=torch.long)
        query_token_type_ids = torch.tensor([b['query_token_type_ids']  for b in batch], dtype=torch.long)

        if self.format == 'pairwise':
            pos_doc_text_emb   = torch.tensor([b['pos_doc_chunk_embeddings'] for b in batch], dtype=torch.float32)
            pos_doc_entity_emb = torch.stack([self._to_entity_tensor(b['pos_doc_ent_emb']) for b in batch])
            neg_doc_text_emb   = torch.tensor([b['neg_doc_chunk_embeddings'] for b in batch], dtype=torch.float32)
            neg_doc_entity_emb = torch.stack([self._to_entity_tensor(b['neg_doc_ent_emb']) for b in batch])

            return {
                'query_input_ids':      query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'pos_doc_text_emb':     pos_doc_text_emb,
                'pos_doc_entity_emb':   pos_doc_entity_emb,
                'neg_doc_text_emb':     neg_doc_text_emb,
                'neg_doc_entity_emb':   neg_doc_entity_emb,
            }

        # Pointwise
        doc_text_emb   = torch.tensor([b['doc_chunk_embeddings'] for b in batch], dtype=torch.float32)
        doc_entity_emb = torch.stack([self._to_entity_tensor(b['doc_ent_emb']) for b in batch])
        label          = torch.tensor([b['label'] for b in batch], dtype=torch.float32)

        res = {
            'query_input_ids':      query_input_ids,
            'query_attention_mask': query_attention_mask,
            'query_token_type_ids': query_token_type_ids,
            'doc_text_emb':         doc_text_emb,
            'doc_entity_emb':       doc_entity_emb,
            'label':                label,
        }
        if not self._train:
            res['query_id'] = [b['query_id'] for b in batch]
            res['doc_id']   = [b['doc_id']   for b in batch]
        return res