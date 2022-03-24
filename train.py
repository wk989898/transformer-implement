import torch
from torch.utils.data import DataLoader
from transformer import Transformer, compute_loss
from datasets import load_dataset
import os
import math
from tqdm import tqdm
import argparse
import random
import numpy as np
from functools import reduce
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from contextlib import ContextDecorator
from functools import wraps

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.processors import TemplateProcessing


class Distributed(ContextDecorator):
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(rank, args):
            self.rank, self.args = rank, args
            with self:
                return fn(rank, args)
        return wrapper

    def __enter__(self):
        rank, args = self.rank, self.args
        dist.init_process_group(
            backend='nccl', init_method=args.init_method, world_size=len(args.gpu_list), rank=rank)
        torch.cuda.set_device(rank)

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.destroy_process_group()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def batch_iterator(dataset):
    for i in range(len(dataset)):
        for key in dataset[i]:
            yield dataset[i][key]


def preprocess(name, tokenizer, max_len=30):
    data_path = f'dataset_{name}_{max_len}.npy'
    if os.path.exists(data_path):
        print(f'{data_path} exists')
        return np.load(data_path)
    dataset = load_dataset(
        'opus100', 'en-zh', split=name).to_dict()['translation']

    def reduce_fn(res, x):
        src, target = tokenizer.encode(x['en']),  tokenizer.encode(x['zh'])
        if 2 < len(src.tokens) < max_len and 2 < len(target.tokens) < max_len:
            src.pad(max_len, pad_id=0)
            target.pad(max_len, pad_id=0)
            res.append((src.ids, target.ids))
        return res
    dataset = reduce(reduce_fn, dataset, [])
    np.save(data_path, dataset)
    return dataset


def loadTokenzier(file='tokenizer.json'):
    if not os.path.exists(file):
        print('train Tokenzier')
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.enable_padding()
        tokenizer.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<PAD>", 0),
                ("<BOS>", 1),
                ("<EOS>", 2),
                ("<UNK>", 3),
            ],
        )
        trainer = BpeTrainer(
            # vocab_size=10000,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"])
        dataset = load_dataset(
            'opus100', 'en-zh', split='train').to_dict()['translation']
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
        tokenizer.save(file)
    else:
        tokenizer = Tokenizer.from_file(file)
    return tokenizer


def update_lr(optimizer, args):
    args.step += 1
    step, warm_step, dim = args.step, args.warm_step, args.dim
    lr = dim**(-0.5)*min(step**(-0.5), step*warm_step**(-1.5))
    for group in optimizer.param_groups:
        group['lr'] = lr


def collate_fn(batch):
    batch = torch.from_numpy(np.array(batch))
    return batch[:, 0, :], batch[:, 1, :]


@Distributed()
def main(rank, args):
    train_data = preprocess('train', tokenizer=args.tokenizer)
    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn)

    validation_data = preprocess('validation', tokenizer=args.tokenizer)
    validation_data = DataLoader(validation_data, batch_size=args.batch_size,
                                 num_workers=args.num_workers, collate_fn=collate_fn)

    args.pad_idx = 0
    args.samples = len(train_data)
    args.step = 0

    model = Transformer(args.vocab_dim, args.dim, args.atten_dim,
                        pad_idx=args.pad_idx, pos_len=args.max_len, recycle=6).cuda()
    if args.check_point is not None:
        model_dict=torch.load(args.check_point)['model']
        model.load_state_dict(model_dict)
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    print(f'args:{args}')
    writer = SummaryWriter(args.log_dir)
    for iter in range(args.epochs):
        total_loss, total_acc, total_n = 0, 0, 0
        model.train()
        for src, target in train_data:
            src, target = src.cuda(), target.cuda()
            label = target[..., 1:]
            target = target[..., :-1]
            optimizer.zero_grad()
            pred = model(src, target)
            loss, acc = compute_loss(
                pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim,smoothing=args.smoothing)
            loss.backward()
            update_lr(optimizer, args)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_n += label.ne(args.pad_idx).sum().item()
        lr = optimizer.param_groups[0]['lr']
        print(
            f'train  iter:{iter} ppl:{math.exp(total_loss/total_n)} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')
        writer.add_scalar('train/loss', total_loss/total_n, iter)
        writer.add_scalar('train/acc', total_acc/total_n, iter)
        writer.add_scalar('train/lr', lr, iter)

        total_loss, total_acc, total_n = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for src, target in validation_data:
                src, target = src.cuda(), target.cuda()
                label = target[..., 1:]
                target = target[..., :-1]
                pred = model(src, target)
                loss, acc = compute_loss(
                    pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim,smoothing=args.smoothing)
                total_loss += loss.item()
                total_acc += acc.item()
                total_n += label.ne(args.pad_idx).sum().item()
        print(
            f'validation iter:{iter} ppl:{math.exp(total_loss/total_n)} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')
        writer.add_scalar('validation/loss', total_loss/total_n, iter)
        writer.add_scalar('validation/acc', total_acc/total_n, iter)
        writer.add_scalar('validation/lr', lr, iter)

    test_data = preprocess('test', tokenizer=args.tokenizer)
    test_data = DataLoader(test_data, batch_size=args.batch_size,
                           num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        total_n, total_acc = 0, 0
        for src, target in test_data:
            src, target = src.cuda(), target.cuda()
            label = target[..., 1:]
            target = target[..., :-1]
            pred = model(src, target)
            non_pad_mask = label.ne(args.pad_idx)
            p = torch.argmax(pred, dim=-1)
            gt = label
            assert p.shape == gt.shape, f'pred shape:{p.shape} and gt shape:{gt.shape}'
            acc = p.eq(gt).masked_select(non_pad_mask).sum()

            total_n += non_pad_mask.sum().item()
            total_acc += acc.item()
        print(f'test acc:{total_acc/total_n:.2f}')

    writer.close()
    if rank == 0:
        model_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'args': args,
            'model': model_dict}, args.save_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('-b', '--batch_size', type=int, default=256)
    parse.add_argument('--max_len', type=int, default=30)
    parse.add_argument('--warm_step', type=int, default=4000)
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)
    parse.add_argument('--smoothing', type=float, default=0.0)

    parse.add_argument('-g', '--gpu_list', nargs='+', type=str)
    parse.add_argument('--init_method', type=str,
                       default='tcp://localhost:23456')
    parse.add_argument('--seed', type=int, help='random seed')
    parse.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader num_workers')
    parse.add_argument('--log_dir', type=str,
                       default='log', help='specify path to save log')
    parse.add_argument('--check_point', type=str,
                       default=None, help='specify check_point path')
    parse.add_argument('--save_path', type=str,
                       default='model.pt', help='specify path to save model')
    args = parse.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [i for i in args.gpu_list])

    args.tokenizer = loadTokenzier()
    args.vocab_dim = args.tokenizer.get_vocab_size()

    mp.spawn(main, nprocs=len(args.gpu_list), args=(args,))
    print('done')
