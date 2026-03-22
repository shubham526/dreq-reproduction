import torch


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, (doc_id, (score, _)) in enumerate(res, start=1):
                writer.write(f'{q_id} Q0 {doc_id} {rank} {score:.6f} DREQ\n')


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins    = int(elapsed / 60)
    secs    = int(elapsed - mins * 60)
    return mins, secs


def save_checkpoint(save_path: str, model):
    """
    Legacy helper — saves model weights only (flat state_dict).
    Prefer saving the full checkpoint dict directly in train.py
    so optimizer/scheduler/epoch/config are included.
    """
    if save_path is None:
        return
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path: str, model, device):
    """
    Load model weights from a checkpoint file.
    Handles both formats:
      - Full checkpoint dict: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'config': ...}
      - Legacy flat state_dict saved directly with torch.save(model.state_dict(), path)

    Returns the full checkpoint dict (or None for legacy format) so callers
    can inspect 'config', 'epoch', 'best_metric', etc.
    """
    if load_path is None:
        return None

    ckpt = torch.load(load_path, map_location=device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        config = ckpt.get('config', {})
        epoch  = ckpt.get('epoch', 0)
        metric = ckpt.get('best_metric', 0.0)
        print(f'Model loaded from <== {load_path}')
        print(f'  Epoch: {epoch}  |  Best metric: {metric:.4f}')
        if config:
            print(f'  Config: {config}')
        return ckpt
    else:
        # Legacy: ckpt IS the state_dict
        model.load_state_dict(ckpt)
        print(f'Model loaded from <== {load_path}  (legacy flat format)')
        return None