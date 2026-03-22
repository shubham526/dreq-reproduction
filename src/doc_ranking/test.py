import torch
import utils
import argparse
import evaluate
from dataset import DocRankingDataset
from model import DocRankingModel
from dataloader import DocRankingDataLoader
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser("Test DREQ document ranking model.")
    parser.add_argument('--test',        help='Test data file.',                          required=True, type=str)
    parser.add_argument('--checkpoint',  help='Model checkpoint to load.',                required=True, type=str)
    parser.add_argument('--save',        help='Output run file in TREC format.',          required=True, type=str)
    parser.add_argument('--text-enc',
                        help='Encoder alias (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). '
                             'Default: bert. Overridden by checkpoint config if available.',
                        type=str, default='bert')
    parser.add_argument('--dropout',
                        help='Dropout rate. Default: 0.1. Overridden by checkpoint config if available.',
                        default=0.1, type=float)
    parser.add_argument('--max-len',     help='Max token length. Default: 512.',  default=512, type=int)
    parser.add_argument('--batch-size',  help='Batch size. Default: 8.',          default=8,   type=int)
    parser.add_argument('--num-workers', help='DataLoader workers. Default: 0.',  default=0,   type=int)
    parser.add_argument('--cuda',        help='CUDA device number. Default: 0.',  default=0,   type=int)
    parser.add_argument('--use-cuda',    help='Use CUDA if available.',           action='store_true')
    parser.add_argument('--dtype',
                        help='AMP dtype (bf16|fp16|fp32). Should match training. Default: bf16.',
                        default='bf16', choices=['bf16', 'fp16', 'fp32'], type=str)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp   = (device.type == 'cuda') and (args.dtype != 'fp32')
    print(f'AMP dtype: {args.dtype}  |  AMP enabled: {use_amp}')

    model_map = {
        'bert':       'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta':    'roberta-base',
        'deberta':    'microsoft/deberta-base',
        'ernie':      'nghuyong/ernie-2.0-base-en',
        'electra':    'google/electra-small-discriminator',
        'conv-bert':  'YituTech/conv-bert-base',
        't5':         't5-base',
    }

    # -------------------------------------------------------------------------
    # Load checkpoint first so we can read pretrained/dropout from config
    # and reconstruct the model exactly as it was trained.
    # -------------------------------------------------------------------------
    print(f'Loading checkpoint from {args.checkpoint}...')
    raw_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    ckpt_config = {}
    if isinstance(raw_ckpt, dict) and 'config' in raw_ckpt:
        ckpt_config = raw_ckpt.get('config', {})
        print(f'Checkpoint config: {ckpt_config}')

    # Resolve pretrained model: checkpoint config > CLI arg
    pretrained_from_ckpt = ckpt_config.get('pretrained')
    if pretrained_from_ckpt:
        pretrain = pretrained_from_ckpt
        print(f'Using pretrained model from checkpoint: {pretrain}')
    else:
        pretrain = model_map.get(args.text_enc, args.text_enc)
        print(f'Using pretrained model from --text-enc: {pretrain}')

    # Resolve dropout: checkpoint config > CLI arg
    dropout_from_ckpt = ckpt_config.get('dropout')
    dropout = dropout_from_ckpt if dropout_from_ckpt is not None else args.dropout
    if dropout_from_ckpt is not None:
        print(f'Using dropout from checkpoint: {dropout}')

    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)

    print('Creating test dataset...')
    test_set = DocRankingDataset(
        dataset=args.test, tokenizer=tokenizer, train=False, max_len=args.max_len
    )

    test_loader = DocRankingDataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # Build model with the same config as training, then load weights
    model = DocRankingModel(pretrained=pretrain, dropout=dropout)
    ckpt = utils.load_checkpoint(args.checkpoint, model, device)
    if ckpt and 'epoch' in ckpt:
        print(f'Checkpoint from epoch {ckpt["epoch"]}  |  '
              f'Best metric: {ckpt.get("best_metric", "n/a")}')
    model.to(device)

    print('Running inference...')
    res_dict = evaluate.evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    print('Writing run file...')
    utils.save_trec(args.save, res_dict)
    print(f'Run file saved to: {args.save}')
    print('Test complete.')


if __name__ == '__main__':
    main()