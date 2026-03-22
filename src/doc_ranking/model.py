import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel

# Dimensions — centralised so they are easy to change for ablations
BERT_HIDDEN_DIM  = 768   # CLS output from BERT / DistilBERT / similar
ENTITY_EMB_DIM   = 300   # Wikipedia2Vec entity embedding size
PROJ_DIM         = 100   # Shared projection dimension for query and doc
# Interaction features: [query, doc, query+doc, query-doc, query*doc] -> 5 * PROJ_DIM
INTERACTION_DIM  = 5 * PROJ_DIM


class TextEmbedding(nn.Module):
    """Wraps a pretrained transformer and returns the CLS (or mean-pool for T5) embedding."""

    def __init__(self, pretrained: str) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        if isinstance(self.encoder, DistilBertModel):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return output[0][:, 0, :]                           # [B, 768]
        elif isinstance(self.encoder, T5EncoderModel):
            last_hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return torch.mean(last_hidden, dim=1)               # [B, 768]
        else:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return output[0][:, 0, :]                           # [B, 768]


class QueryEmbedding(nn.Module):
    """
    Encodes a query via a pretrained transformer then projects to PROJ_DIM.
    Dropout is applied before the projection to regularise the BERT representation.
    """

    def __init__(self, pretrained: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TextEmbedding(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(BERT_HIDDEN_DIM, PROJ_DIM)

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        text_emb = self.encoder(input_ids, attention_mask, token_type_ids)  # [B, 768]
        return self.fc(self.dropout(text_emb))                               # [B, 100]


class DocEmbedding(nn.Module):
    """
    Combines the mean-pooled chunk embedding (768-d) and the summed entity
    embedding (300-d) into a single PROJ_DIM vector.
    Dropout is applied to the concatenated representation before projection.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(BERT_HIDDEN_DIM + ENTITY_EMB_DIM, PROJ_DIM)

    def forward(self, text_emb: torch.Tensor, entity_emb: torch.Tensor) -> torch.Tensor:
        # text_emb:   [B, 768]
        # entity_emb: [B, 300]  (zero vector when no entity overlap)
        concat = torch.cat((text_emb, entity_emb), dim=1)      # [B, 1068]
        return self.fc(self.dropout(concat))                    # [B, 100]


class DocRankingModel(nn.Module):
    """
    DREQ scoring model.

    Query and document are each projected to PROJ_DIM (100-d).
    Five interaction features are computed (identity, sum, difference, product)
    and concatenated -> INTERACTION_DIM (500-d) -> scalar relevance score.

    Dropout before the final scoring layer prevents overfitting on small
    collections like Robust04.
    """

    def __init__(self, pretrained: str, dropout: float = 0.1):
        super().__init__()
        self._dropout_rate = dropout          # stored for checkpoint config serialisation
        self.query_encoder = QueryEmbedding(pretrained=pretrained, dropout=dropout)
        self.doc_encoder   = DocEmbedding(dropout=dropout)
        self.dropout       = nn.Dropout(dropout)
        # 5 * PROJ_DIM = 500 -> 1
        self.score = nn.Linear(INTERACTION_DIM, 1)

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        doc_text_emb,
        doc_entity_emb,
    ):
        query_emb = self.query_encoder(query_input_ids, query_attention_mask, query_token_type_ids)  # [B, 100]
        doc_emb   = self.doc_encoder(doc_text_emb, doc_entity_emb)                                   # [B, 100]

        # Interaction features
        emb_add = query_emb + doc_emb           # element-wise sum
        emb_sub = query_emb - doc_emb           # element-wise difference
        emb_mul = query_emb * doc_emb           # element-wise product

        # Concatenate all five views: [B, 500]
        interaction = torch.cat((query_emb, doc_emb, emb_add, emb_sub, emb_mul), dim=1)

        score = self.score(self.dropout(interaction))   # [B, 1]
        return score.squeeze(dim=1)                     # [B]