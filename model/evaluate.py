import torch
from tqdm import tqdm


def evaluate(model, data_loader, device, use_amp=False, amp_dtype=torch.bfloat16):
    """
    Run inference over data_loader and return a result dict.

    Args:
        model:      DocRankingModel (already on device, set to eval externally or here).
        data_loader: DataLoader over a non-train DocRankingDataset.
        device:     torch.device.
        use_amp:    Enable autocast during inference (matches training precision).
        amp_dtype:  AMP dtype — should match what was used during training.

    Returns:
        {query_id: {doc_id: [score, label]}}
    """
    rst_dict = {}
    model.eval()

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for batch in tqdm(data_loader, desc="Evaluating"):
                query_id = batch['query_id']
                doc_id   = batch['doc_id']
                label    = batch['label']

                batch_score = model(
                    query_input_ids      = batch['query_input_ids'].to(device),
                    query_attention_mask = batch['query_attention_mask'].to(device),
                    query_token_type_ids = batch['query_token_type_ids'].to(device),
                    doc_text_emb         = batch['doc_text_emb'].to(device),
                    doc_entity_emb       = batch['doc_entity_emb'].to(device),
                )

                batch_score = batch_score.detach().cpu().tolist()

                for q_id, d_id, score, lbl in zip(query_id, doc_id, batch_score, label):
                    if q_id not in rst_dict:
                        rst_dict[q_id] = {}
                    # Keep highest score if doc appears more than once (shouldn't happen normally)
                    if d_id not in rst_dict[q_id] or score > rst_dict[q_id][d_id][0]:
                        rst_dict[q_id][d_id] = [score, lbl]

    return rst_dict