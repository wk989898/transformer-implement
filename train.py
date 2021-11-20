import torch
from torch.utils.data import DataLoader, Dataset
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

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.processors import TemplateProcessing


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def batch_iterator(dataset):
    for i in range(len(dataset)):
        yield dataset[i]["cs"]
        yield dataset[i]["en"]


def preprocess(name, tokenizer=None, file='tokenizer.json'):
    data_path = f'dataset_{name}_{args.max_len}.npy'
    if os.path.exists(file) and os.path.exists(data_path):
        print(f'{data_path} exists')
        return np.load(data_path), Tokenizer.from_file(file) if tokenizer is None else tokenizer

    dataset = load_dataset(
        'wmt16', 'cs-en', split=name).to_dict()['translation']
    if tokenizer is None:
        assert name == 'train', 'tokenizer must use train_data'
        tokenizer = loadTokenzier(dataset)

    def reduce_fn(res, x):
        cs, en = tokenizer.encode(x['cs']), tokenizer.encode(x['en'])
        if 2 < len(cs.tokens) < args.max_len and 2 < len(en.tokens) < args.max_len:
            cs.pad(args.max_len, pad_id=0)
            en.pad(args.max_len, pad_id=0)
            res.append((cs.ids, en.ids))
        return res
    dataset = reduce(reduce_fn, dataset, [])
    np.save(data_path, dataset)
    return (dataset, tokenizer) if tokenizer is not None else dataset


def loadTokenzier(dataset, file='tokenizer.json'):
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

    cs, en = torch.from_numpy(np.array(batch)).chunk(2, dim=1)
    return cs.squeeze(), en.squeeze()


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in args.gpu_list])

    train_data, tokenizer = preprocess('train')
    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn)

    validation_data = preprocess('validation', tokenizer=tokenizer)
    validation_data = DataLoader(validation_data, batch_size=args.batch_size,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    args.vocab_dim = tokenizer.get_vocab_size()
    args.pad_idx = 0
    args.samples = len(train_data)
    args.step = 0

    model = Transformer(args.vocab_dim, args.dim, args.atten_dim,
                        pad_idx=args.pad_idx, pos_len=args.max_len, recycle=6).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    if len(args.gpu_list) > 1:
        model = torch.nn.DataParallel(model)
    print(f'args:{args}')
    writer = SummaryWriter(args.log_dir)
    for iter in range(args.epochs):
        total_loss, total_acc, total_n = 0, 0, 0
        model.train()
        for cs, en in train_data:
            cs, en = cs.cuda(), en.cuda()
            label = en[..., 1:]
            en = en[..., :-1]
            optimizer.zero_grad()
            pred = model(cs, en)
            loss, acc = compute_loss(
                pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)
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
            for cs, en in validation_data:
                cs, en = cs.cuda(), en.cuda()
                label = en[..., 1:]
                en = en[..., :-1]
                pred = model(cs, en)
                loss, acc = compute_loss(
                    pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)
                total_loss += loss.item()
                total_acc += acc.item()
                total_n += label.ne(args.pad_idx).sum().item()
        print(
            f'validation iter:{iter} ppl:{math.exp(total_loss/total_n)} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}\n')
        writer.add_scalar('validation/loss', total_loss/total_n, iter)
        writer.add_scalar('validation/acc', total_acc/total_n, iter)
        writer.add_scalar('validation/lr', lr, iter)
    test_data = preprocess('test', tokenizer=tokenizer)
    test_data = DataLoader(test_data, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        total_n, total_acc = 0, 0
        for cs, en in test_data:
            cs, en = cs.cuda(), en.cuda()
            label = en[..., 1:]
            en = en[..., :-1]
            pred = model(cs, en)
            non_pad_mask = label.ne(args.pad_idx)
            p = torch.argmax(pred, dim=-1)
            gt = label
            assert p.shape == gt.shape, f'pred shape:{p.shape} and gt shape:{gt.shape}'
            acc = p.eq(gt).masked_select(non_pad_mask).sum()

            total_n += non_pad_mask.sum().item()
            total_acc += acc.item()
        print(f'acc:{total_acc/total_n:.2f}')

    writer.close()
    save_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'vocab_dim': args.vocab_dim,
        'dim': args.dim,
        'atten_dim': args.atten_dim,
        'model': save_model.state_dict()
    }, args.save_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('-b', '--batch_size', type=int, default=256)
    parse.add_argument('--max_len', type=int, default=30)
    parse.add_argument('--warm_step', type=int, default=4000)
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)

    parse.add_argument('-g', '--gpu_list', nargs='+', type=int)
    parse.add_argument('--seed', type=int, help='random seed')
    parse.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader num_workers')
    parse.add_argument('--log_dir', type=str,
                       default='log', help='specify path to save log')
    parse.add_argument('--save_path', type=str,
                       default='model.pt', help='specify path to save model')
    args = parse.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    main(args)
    print('done')
