import torch
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer, compute_loss
from datasets import load_dataset
import os
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
        yield dataset[i]["cs"]
        yield dataset[i]["en"]


def preprocess(dataset, file='tokenizer.json', prefix='', tokenizer=None):
    if tokenizer is None:
        tokenizer = loadTokenzier(dataset, file, prefix=prefix)

    def reduce_fn(res, x):
        cs, en = tokenizer.encode(x['cs']), tokenizer.encode(x['en'])
        if 2 < len(cs.tokens) < args.max_len and 2 < len(en.tokens) < args.max_len:
            cs.pad(args.max_len, pad_id=0)
            en.pad(args.max_len, pad_id=0)
            res.append((cs.ids, en.ids))
        return res
    dataset = reduce(reduce_fn, dataset, [])

    return dataset, tokenizer


def loadTokenzier(dataset, file='tokenizer.json', prefix=''):
    if not os.path.exists(prefix+file):
        print('train Tokenzier')
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.enable_padding()
        tokenizer.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()
        # tokenizer.post_processor = TemplateProcessing(
        #     single="[CLS] $A [SEP]",
        #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        #     special_tokens=[
        #         ("[CLS]", tokenizer.token_to_id("[CLS]")),
        #         ("[SEP]", tokenizer.token_to_id("[SEP]")),
        #     ],
        # )
        trainer = BpeTrainer(
            # vocab_size=10000,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
        tokenizer.save(prefix+file)
    else:
        tokenizer = Tokenizer.from_file(prefix+file)

    return tokenizer


def update_lr(optimizer, args):
    args.step += 1
    step, warm_step, dim = args.step, args.warm_step, args.dim
    lr = dim**(-0.5)*min(step**(-0.5), step*warm_step**(-1.5))
    for group in optimizer.param_groups:
        group['lr'] = lr


def collate_fn(batch):
    cs, en = torch.tensor(batch).cuda().chunk(2, dim=1)
    return cs.squeeze(), en.squeeze()


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in args.gpu_list])

    dataset = load_dataset(
        'wmt16', 'cs-en', split='train').to_dict()['translation']
    train_data, tokenizer = preprocess(dataset)
    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=0, shuffle=True, collate_fn=collate_fn)

    validation = load_dataset(
        'wmt16', 'cs-en', split="validation").to_dict()['translation']
    validation_data, _ = preprocess(
        validation, prefix='validation-', tokenizer=tokenizer)
    validation_data = DataLoader(validation_data, batch_size=args.batch_size,
                                 num_workers=0, shuffle=True, collate_fn=collate_fn)

    args.vocab_dim = tokenizer.get_vocab_size()
    args.pad_idx = 0
    args.samples = len(train_data)
    args.step = 0

    model = Transformer(args.vocab_dim, args.dim,
                        args.atten_dim, pad_idx=args.pad_idx, recycle=6)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    if len(args.gpu_list) > 1:
        model = torch.nn.DataParallel(model).cuda()
    print(f'args:{args}')
    writer = SummaryWriter(args.log_dir)
    for iter in range(args.epochs):
        total_loss, total_acc, total_n = 0, 0, 0
        model.train()
        for cs, en in train_data:
            label = en[..., 1:].clone()
            en = en[..., :-1]
            pred = model(cs, en)
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
            f'train  iter:{iter} loss:{total_loss/total_n} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')
        writer.add_scalar('loss', total_loss/total_n)
        writer.add_scalar('acc', total_acc/total_n)
        writer.add_scalar('lr', lr)

        total_loss, total_acc, total_n = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for cs, en in validation_data:
                label = en[..., 1:].clone()
                en = en[..., :-1]
                pred = model(cs, en)
                loss, acc = compute_loss(
                    pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)
                total_loss += loss.item()
                total_acc += acc.item()
                total_n += label.ne(args.pad_idx).sum().item()
        print(
            f'validation iter:{iter} loss:{total_loss/total_n} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')

    test = load_dataset(
        'wmt16', 'cs-en', split="test").to_dict()['translation']
    test_data, _ = preprocess(
        test, prefix='test-', tokenizer=tokenizer)
    test_data = DataLoader(test_data, batch_size=args.batch_size,
                           num_workers=0, shuffle=True, collate_fn=collate_fn)
    model.eval()
    with torch.no_grad():
        total_n, total_acc = 0, 0
        for cs, en in test_data:
            label = en[..., 1:].clone()
            en = en[..., :-1]
            pred = model(cs, en)
            non_pad_mask = label.ne(0)
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
    parse.add_argument('--batch_size', type=int, default=256)
    parse.add_argument('--max_len', type=int, default=30)
    parse.add_argument('--warm_step', type=int, default=200000)
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)

    parse.add_argument('-g', '--gpu_list', nargs='+', type=int)
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
