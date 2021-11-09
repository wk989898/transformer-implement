import torch
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer, compute_loss
from datasets import load_dataset
import os
from tqdm import tqdm
import argparse
import random

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
    dataset = map(lambda x: (tokenizer.encode(
        x['cs']), tokenizer.encode(x['en'])), dataset)
    dataset = filter(lambda x: 2 < len(
        x[0].tokens) < args.max_len and 2 < len(x[1].tokens) < args.max_len, dataset)

    return list(dataset), tokenizer


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


def update_lr(optimizer, step, args):
    warm_step, dim = args.warm_step, args.dim
    lr = (dim**(-0.5))*min(step**(-0.5), step*warm_step**(-1.5))
    for group in optimizer.param_groups:
        group['lr'] = lr


def collate_fn(batch):
    cs_batch, en_batch = [], []
    for cs, en in batch:
        cs.pad(args.max_len, pad_id=0)
        en.pad(args.max_len, pad_id=0)
        cs_batch.append(cs.ids)
        en_batch.append(en.ids)
    return cs_batch, en_batch


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = load_dataset(
        'wmt16', 'cs-en', split='train').to_dict()['translation']
    train_data, tokenizer = preprocess(dataset)
    args.vocab_dim = tokenizer.get_vocab_size()
    args.pad_idx = 0

    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=8, collate_fn=collate_fn)

    model = Transformer(args.vocab_dim, args.dim,
                        args.atten_dim, pad_idx=args.pad_idx, recycle=7)
    device = 'cpu'
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = 'cuda'
        model = torch.nn.DataParallel(model).to(device)
    print(f'args:{args}')

    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    model.train()
    for iter in range(args.epoch):
        total_loss = 0
        total_acc = 0
        total_n = 0
        for cs, en in tqdm(train_data):
            cs, en = torch.tensor(cs).to(device), torch.tensor(en).to(device)
            label = en[..., 1:].clone()
            en = en[..., :-1]
            pred = model(cs, en)
            loss, acc = compute_loss(
                pred, label, pad_idx=args.pad_idx, vocab_dim=args.vocab_dim)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_lr(optimizer, iter+1, args)

            total_loss += loss.item()
            total_acc += acc.item()
            total_n += label.ne(args.pad_idx).sum().item()
        lr = optimizer.param_groups[0]['lr']
        print(
            f'iter:{iter} loss:{total_loss/total_n} acc:{total_acc/total_n} total_words:{total_n} lr:{lr}')

    validation = load_dataset(
        'wmt16', 'cs-en', split="validation").to_dict()['translation']
    validation_data, _ = preprocess(
        validation, prefix='validation-', tokenizer=tokenizer)

    model.eval()
    with torch.no_grad():
        for cs, en in tqdm(validation_data):
            cs, en = torch.tensor(cs).to(device), torch.tensor(en).to(device)
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

    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)
    parse.add_argument('--epoch', type=int, default=1000)
    parse.add_argument('--warm_step', type=int, default=4000)
    parse.add_argument('--max_len', type=int, default=40)
    parse.add_argument('--batch_size', type=int, default=512)
    parse.add_argument('--seed', type=int)
    parse.add_argument('--save_path', type=str,
                       default='model.pt', help='specify path to save model')
    args = parse.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    main(args)
    print('done')
