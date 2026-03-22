import os
import time
import json
import shutil
import torch
import torch.nn as nn
import utils
import metrics
import argparse
import evaluate
import random
import numpy as np
from dataset import DocRankingDataset
from model import DocRankingModel
from trainer import Trainer
from dataloader import DocRankingDataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save,
          run_file, eval_every, device, patience, use_amp, amp_dtype, start_epoch=0,
          best_metric_so_far=0.0):
    best_valid_metric = best_metric_so_far
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_metric': [], 'epoch': []}

    history_path = os.path.join(save_path, 'training_history.json')
    if start_epoch > 0 and os.path.exists(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
            print(f'Loaded training history (up to epoch {start_epoch})')
        except Exception:
            print('Could not load training history, starting fresh.')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'{"=" * 60}')

        train_loss = trainer.train()
        history['train_loss'].append(train_loss)
        history['epoch'].append(epoch + 1)

        if (epoch + 1) % eval_every == 0:
            print('Running validation...')
            model.eval()
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    res_dict = evaluate.evaluate(model=model, data_loader=valid_loader, device=device)

            run_filename = f'epoch_{epoch + 1:03d}_{run_file}'
            run_path = os.path.join(save_path, run_filename)
            utils.save_trec(run_path, res_dict)

            valid_metric = metrics.get_metric(qrels, run_path, metric)
            history['val_metric'].append(valid_metric)

            improvement = valid_metric - best_valid_metric
            epoch_mins, epoch_secs = utils.epoch_time(start_time, time.time())

            print(f'\n{"─" * 60}')
            print(f'Epoch Time  : {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss  : {train_loss:.4f}')
            print(f'Val {metric.upper():<8}: {valid_metric:.4f}')
            print(f'Best {metric.upper():<7}: {best_valid_metric:.4f}')
            print(f'{"─" * 60}')

            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                epochs_without_improvement = 0

                checkpoint = {
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'epoch':                epoch + 1,
                    'best_metric':          best_valid_metric,
                }
                # Save named checkpoint + overwrite the canonical best checkpoint
                named_path = os.path.join(save_path, f'best_model_epoch_{epoch + 1:03d}.bin')
                torch.save(checkpoint, named_path)
                torch.save(checkpoint, os.path.join(save_path, save))
                shutil.copy(run_path, os.path.join(save_path, f'best_{run_file}'))

                print(f'New best {metric.upper()}: {best_valid_metric:.4f} (↑ {improvement:.4f})')
                print(f'Saved checkpoint: {named_path}')
            else:
                epochs_without_improvement += 1
                direction = '↓' if improvement < 0 else '='
                print(f'No improvement {metric.upper()}: {valid_metric:.4f} '
                      f'({direction} {abs(improvement):.4f}) '
                      f'[{epochs_without_improvement}/{patience}]')

                if patience > 0 and epochs_without_improvement >= patience:
                    print(f'\nEarly stopping after {epoch + 1} epochs. '
                          f'Best {metric.upper()}: {best_valid_metric:.4f}')
                    break

    # Finalise history
    history['best_epoch'] = (
        history['epoch'][history['val_metric'].index(max(history['val_metric']))]
        if history['val_metric'] else 0
    )
    history['best_val_metric']   = max(history['val_metric']) if history['val_metric'] else 0
    history['final_train_loss']  = history['train_loss'][-1]  if history['train_loss']  else 0

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nTraining history saved to {history_path}')

    return best_valid_metric


def main():
    parser = argparse.ArgumentParser("Train DREQ document ranking doc_ranking.")
    parser.add_argument('--train',          help='Training data file.',          required=True,  type=str)
    parser.add_argument('--dev',            help='Validation data file.',         required=True,  type=str)
    parser.add_argument('--qrels',          help='Qrels file in TREC format.',    required=True,  type=str)
    parser.add_argument('--save-dir',       help='Directory to save outputs.',    required=True,  type=str)
    parser.add_argument('--save',           help='Checkpoint filename.',          default='doc_ranking.bin',  type=str)
    parser.add_argument('--checkpoint',     help='Checkpoint to resume from.',    default=None,         type=str)
    parser.add_argument('--run',            help='Validation run filename.',      default='dev.run',    type=str)
    parser.add_argument('--text-enc',
                        help='Encoder (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). '
                             'Default: bert.',
                        type=str, default='bert')
    parser.add_argument('--metric',         help='Validation metric. Default: map.',    default='map',  type=str)
    parser.add_argument('--max-len',        help='Max token length. Default: 512.',     default=512,    type=int)
    parser.add_argument('--epoch',          help='Max epochs. Default: 20.',            default=20,     type=int)
    parser.add_argument('--batch-size',     help='Batch size. Default: 8.',             default=8,      type=int)
    parser.add_argument('--learning-rate',  help='Learning rate. Default: 2e-5.',       default=2e-5,   type=float)
    parser.add_argument('--n-warmup-steps', help='LR warmup steps. Default: 1000.',     default=1000,   type=int)
    parser.add_argument('--eval-every',     help='Evaluate every N epochs. Default: 1.',default=1,      type=int)
    parser.add_argument('--patience',
                        help='Early stopping patience. 0 = disabled. Default: 5.',
                        default=5, type=int)
    parser.add_argument('--dropout',        help='Dropout rate. Default: 0.1.',         default=0.1,    type=float)
    parser.add_argument('--num-workers',    help='DataLoader workers. Default: 0.',      default=0,     type=int)
    parser.add_argument('--cuda',           help='CUDA device number. Default: 0.',      default=0,     type=int)
    parser.add_argument('--use-cuda',       help='Use CUDA if available.',               action='store_true')
    parser.add_argument('--dtype',
                        help='Precision for AMP (bf16|fp16|fp32). Default: bf16.',
                        default='bf16', choices=['bf16', 'fp16', 'fp32'], type=str)
    parser.add_argument('--random-seed',    help='Random seed. Default: 42.',            default=42,    type=int)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------
    # Device + Ampere optimisations
    # -------------------------------------------------------------------------
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    if device.type == 'cuda':
        # TF32 gives ~3x speedup on Ampere matmuls with negligible accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp   = (device.type == 'cuda') and (args.dtype != 'fp32')

    # -------------------------------------------------------------------------
    # Config summary
    # -------------------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('DREQ Training Configuration')
    print(f'{"=" * 60}')
    print(f'Device      : {device}')
    print(f'AMP dtype   : {args.dtype}  |  AMP enabled: {use_amp}')
    print(f'Encoder     : {args.text_enc}')
    print(f'LR          : {args.learning_rate}')
    print(f'Batch size  : {args.batch_size}')
    print(f'Dropout     : {args.dropout}')
    print(f'Patience    : {args.patience}')
    print(f'Random seed : {args.random_seed}')
    if args.checkpoint:
        print(f'Resuming from: {args.checkpoint}')
    print(f'{"=" * 60}\n')

    os.makedirs(args.save_dir, exist_ok=True)

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
    pretrain  = model_map[args.text_enc]
    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)

    # Save run config
    config = {
        'max_len': args.max_len, 'doc_ranking': pretrain, 'metric': args.metric,
        'epochs': args.epoch, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'warmup_steps': args.n_warmup_steps,
        'dropout': args.dropout, 'patience': args.patience,
        'dtype': args.dtype, 'random_seed': args.random_seed,
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('Creating datasets...')
    train_set = DocRankingDataset(dataset=args.train, tokenizer=tokenizer, train=True,  max_len=args.max_len)
    dev_set   = DocRankingDataset(dataset=args.dev,   tokenizer=tokenizer, train=False, max_len=args.max_len)
    print(f'Train format detected: {train_set.format.upper()}')

    # pin_memory + persistent_workers speed up GPU data transfers
    train_loader = DocRankingDataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    dev_loader = DocRankingDataLoader(
        dataset=dev_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    model   = DocRankingModel(pretrained=pretrain, dropout=args.dropout)
    loss_fn = (
        nn.MarginRankingLoss(margin=1.0)
        if train_set.format == 'pairwise'
        else nn.BCEWithLogitsLoss()
    )
    print(f'Loss function: {loss_fn.__class__.__name__}')
    print(f'Total parameters    : {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------
    start_epoch       = 0
    best_metric_so_far = 0.0
    optimizer_state   = None
    scheduler_state   = None

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f'Checkpoint not found: {args.checkpoint} — starting from scratch.')
        else:
            try:
                ckpt = torch.load(args.checkpoint, map_location='cpu')
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    optimizer_state    = ckpt.get('optimizer_state_dict')
                    scheduler_state    = ckpt.get('scheduler_state_dict')
                    start_epoch        = ckpt.get('epoch', 0)
                    best_metric_so_far = ckpt.get('best_metric', 0.0)
                    print(f'Resuming from epoch {start_epoch}, best {args.metric}: {best_metric_so_far:.4f}')
                else:
                    model.load_state_dict(ckpt)
                    print('Loaded weights (legacy format).')
            except Exception as e:
                print(f'Error loading checkpoint: {e} — starting from scratch.')

    model.to(device)
    loss_fn.to(device)

    # Separate weight decay: don't apply to bias / LayerNorm params
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print('Loaded optimizer state.')
        except Exception as e:
            print(f'Could not load optimizer state: {e}')

    num_training_steps = (len(train_set) // args.batch_size) * args.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print('Loaded scheduler state.')
        except Exception as e:
            print(f'Could not load scheduler state: {e}')

    # -------------------------------------------------------------------------
    # Sanity check — one forward pass before committing to full training
    # -------------------------------------------------------------------------
    print('\nRunning sanity check...')
    try:
        test_batch = next(iter(train_loader))
        batch_format = test_batch.get('format', 'pointwise')
        model.eval()
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if batch_format == 'pairwise':
                    scores = model(
                        query_input_ids      = test_batch['query_input_ids'].to(device),
                        query_attention_mask = test_batch['query_attention_mask'].to(device),
                        query_token_type_ids = test_batch['query_token_type_ids'].to(device),
                        doc_text_emb         = test_batch['pos_doc_text_emb'].to(device),
                        doc_entity_emb       = test_batch['pos_doc_entity_emb'].to(device),
                    )
                else:
                    scores = model(
                        query_input_ids      = test_batch['query_input_ids'].to(device),
                        query_attention_mask = test_batch['query_attention_mask'].to(device),
                        query_token_type_ids = test_batch['query_token_type_ids'].to(device),
                        doc_text_emb         = test_batch['doc_text_emb'].to(device),
                        doc_entity_emb       = test_batch['doc_entity_emb'].to(device),
                    )
        print(f'Sanity check passed ({batch_format.upper()}) — '
              f'score range [{scores.min():.3f}, {scores.max():.3f}]')
    except Exception as e:
        import traceback
        print(f'Sanity check FAILED: {e}')
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=loss_fn,
        scheduler=scheduler, data_loader=train_loader, device=device,
        use_amp=use_amp, amp_dtype=amp_dtype,
    )

    best_metric = train(
        model=model, trainer=trainer, epochs=args.epoch,
        metric=args.metric, qrels=args.qrels,
        valid_loader=dev_loader, save_path=args.save_dir,
        save=args.save, run_file=args.run,
        eval_every=args.eval_every, device=device,
        patience=args.patience,
        use_amp=use_amp, amp_dtype=amp_dtype,
        start_epoch=start_epoch, best_metric_so_far=best_metric_so_far,
    )

    print(f'\n{"=" * 60}')
    print(f'Training complete.')
    print(f'Best {args.metric.upper()}: {best_metric:.4f}')
    print(f'Model saved to: {os.path.join(args.save_dir, args.save)}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()