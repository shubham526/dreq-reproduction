import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, data_loader, device,
                 max_grad_norm: float = 1.0,
                 use_amp: bool = False,
                 amp_dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            model:          DocRankingModel instance.
            optimizer:      AdamW.
            criterion:      BCEWithLogitsLoss (pointwise) or MarginRankingLoss (pairwise).
            scheduler:      LR scheduler.
            data_loader:    Training DataLoader.
            device:         torch.device.
            max_grad_norm:  Gradient clipping threshold. 1.0 is standard for BERT fine-tuning.
            use_amp:        Enable automatic mixed precision.
            amp_dtype:      torch.bfloat16 (Ampere, no scaler needed) or torch.float16.
        """
        self.model         = model
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.scheduler     = scheduler
        self.data_loader   = data_loader
        self.device        = device
        self.max_grad_norm = max_grad_norm
        self.use_amp       = use_amp
        self.amp_dtype     = amp_dtype

        # GradScaler only needed for fp16 — bf16 has wide enough dynamic range
        # and doesn't need loss scaling. Scaler is a no-op when enabled=False.
        self._scaler = torch.amp.GradScaler(
            'cuda',
            enabled=(use_amp and amp_dtype == torch.float16)
        )

    def train(self) -> float:
        """Run one full epoch. Returns mean loss over non-skipped batches."""
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        nan_batches = 0

        for batch in tqdm(self.data_loader, desc="Training"):
            self.optimizer.zero_grad()

            batch_format = batch.get('format', 'pointwise')

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):

                if batch_format == 'pairwise':
                    pos_score = self.model(
                        query_input_ids      = batch['query_input_ids'].to(self.device),
                        query_attention_mask = batch['query_attention_mask'].to(self.device),
                        query_token_type_ids = batch['query_token_type_ids'].to(self.device),
                        doc_text_emb         = batch['pos_doc_text_emb'].to(self.device),
                        doc_entity_emb       = batch['pos_doc_entity_emb'].to(self.device),
                    )
                    neg_score = self.model(
                        query_input_ids      = batch['query_input_ids'].to(self.device),
                        query_attention_mask = batch['query_attention_mask'].to(self.device),
                        query_token_type_ids = batch['query_token_type_ids'].to(self.device),
                        doc_text_emb         = batch['neg_doc_text_emb'].to(self.device),
                        doc_entity_emb       = batch['neg_doc_entity_emb'].to(self.device),
                    )

                    if self._has_nan(pos_score) or self._has_nan(neg_score):
                        nan_batches += 1
                        continue

                    target = torch.ones_like(pos_score)
                    loss   = self.criterion(pos_score, neg_score, target)

                else:  # pointwise
                    score = self.model(
                        query_input_ids      = batch['query_input_ids'].to(self.device),
                        query_attention_mask = batch['query_attention_mask'].to(self.device),
                        query_token_type_ids = batch['query_token_type_ids'].to(self.device),
                        doc_text_emb         = batch['doc_text_emb'].to(self.device),
                        doc_entity_emb       = batch['doc_entity_emb'].to(self.device),
                    )

                    if self._has_nan(score):
                        nan_batches += 1
                        continue

                    loss = self.criterion(score, batch['label'].to(self.device))

            if torch.isnan(loss):
                nan_batches += 1
                continue

            # Scale → backward → unscale → clip → step
            # (scaler is a no-op for bf16 / fp32 — safe to call unconditionally)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.scheduler.step()

            total_loss  += loss.item()
            num_batches += 1

        if nan_batches > 0:
            print(f'Warning: {nan_batches}/{len(self.data_loader)} batches skipped (NaN/Inf).')

        return total_loss / max(num_batches, 1)

    @staticmethod
    def _has_nan(tensor: torch.Tensor) -> bool:
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()