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
    if save_path is None:
        return
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path: str, model, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f'Model loaded from <== {load_path}')