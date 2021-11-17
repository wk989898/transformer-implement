import torch
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer, compute_loss
from datasets import load_dataset
import os
import math
from tqdm import tqdm
import argparse
import random
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
        yield dataset[i]["zh"]
        yield dataset[i]["en"]


def preprocess(dataset, file='tokenizer.json', tokenizer=None):
    if tokenizer is None:
        tokenizer = loadTokenzier(dataset, file)

    def reduce_fn(res, x):
        zh, en = tokenizer.encode(x['zh']), tokenizer.encode(x['en'])
        if 2 < len(zh.tokens) < args.max_len and 2 < len(en.tokens) < args.max_len:
            zh.pad(args.max_len, pad_id=0)
            en.pad(args.max_len, pad_id=0)
            res.append((zh.ids, en.ids))
        return res
    dataset = reduce(reduce_fn, dataset, [])

    return dataset, tokenizer


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
            vocab_size=50000,
            special_tokens=["<PAD>", "<BOS>", "<EOS>","<UNK>"])
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
    '''
    [batch (a,b)]
    '''
    zh,en=torch.tensor(batch).chunk(2,dim=1)
    return zh.squeeze(),en.squeeze()


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(x) for x in args.gpu_list])

    dataset = load_dataset(
        'opus100', 'en-zh', split='train').to_dict()['translation']
    train_data, tokenizer = preprocess(dataset)

    validation = load_dataset(
        'opus100', 'en-zh', split="validation").to_dict()['translation']
    validation_data, _ = preprocess(validation,  tokenizer=tokenizer)

    args.vocab_dim = tokenizer.get_vocab_size()
    args.pad_idx = 0
    args.step = 0
    args.samples = len(train_data)

    model = Transformer(args.vocab_dim, args.dim,
                        args.atten_dim, pad_idx=args.pad_idx, recycle=6).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    if len(args.gpu_list)>1:
        model = torch.nn.DataParallel(model)
    print(f'args:{args}')
    writer = SummaryWriter(args.log_dir)
    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=8, shuffle=True, collate_fn=collate_fn)
    validation_data = DataLoader(validation_data, batch_size=args.batch_size,
                                 num_workers=8,collate_fn=collate_fn)
    for iter in range(args.epochs):
        total_loss, total_acc, total_n = 0, 0, 0
        model.train()
        for zh, en in train_data:
            zh, en =zh.cuda(), en.cuda()
            label = en[..., 1:]
            en = en[..., :-1]
            pred = model(zh, en)
            loss, acc = compute_loss(
                pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)
            loss.backward()
            update_lr(optimizer, args)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_acc += acc.item()
            total_n += label.ne(args.pad_idx).sum().item()
        lr = optimizer.param_groups[0]['lr']
        print(
            f'train  iter:{iter} ppl:{math.exp(total_loss/total_n)} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')
        writer.add_scalar('loss', total_loss/total_n)
        writer.add_scalar('acc', total_acc/total_n)
        writer.add_scalar('lr', lr)

        total_loss, total_acc, total_n = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for zh, en in validation_data:
                zh, en =zh.cuda(), en.cuda()
                label = en[..., 1:]
                en = en[..., :-1]
                pred = model(zh, en)
                loss, acc = compute_loss(
                    pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)

                total_loss += loss.item()
                total_acc += acc.item()
                total_n += label.ne(args.pad_idx).sum().item()
            print(
                f'validation  iter:{iter} ppl:{math.exp(total_loss/total_n)} acc:{total_acc/total_n} total_words:{total_n}')

    test = load_dataset(
        'opus100', 'en-zh', split="test").to_dict()['translation']
    test_data, _ = preprocess(
        test,  tokenizer=tokenizer)
    test_data = DataLoader(test_data, batch_size=args.batch_size,
                           num_workers=8,collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        total_acc, total_n = 0, 0
        for zh, en in test_data:
            zh, en =zh.cuda(), en.cuda()
            label = en[..., 1:]
            en = en[..., :-1]
            pred = model(zh, en)
            non_pad_mask = label.ne(args.pad_idx)
            p = torch.argmax(pred, dim=-1)
            gt = label
            assert p.shape == gt.shape, f'pred shape:{p.shape} and gt shape:{gt.shape}'
            acc = p.eq(gt).masked_select(non_pad_mask).sum()

            total_n += non_pad_mask.sum().item()
            total_acc += acc.item()
        print(f'acc:{total_acc/total_n:.2f}')

    writer.close()
    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('-b','--batch_size', type=int, default=256)
    parse.add_argument('--max_len', type=int, default=30)
    parse.add_argument('--warm_step', type=int, default=40000)
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)

    parse.add_argument('-g', '--gpu_list', nargs='+', type=int)
    parse.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader num workers')
    parse.add_argument('--seed', type=int, help='random seed')
    parse.add_argument('--log_dir', type=str,
                       default='log', help='specify path to save log')
    parse.add_argument('--save_path', type=str,
                       default='model.pt', help='specify path to save model')
    args = parse.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    main(args)
    print('done')
