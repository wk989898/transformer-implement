import torch
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer
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
        tokenizer = loadTokenzier(dataset, file)
    dataset = map(lambda x: (tokenizer.encode(
        x['cs']), tokenizer.encode(x['en'])), dataset)
    dataset = filter(lambda x: 2 < len(
        x[0].tokens) < args.max_len and 2 < len(x[1].tokens) < args.max_len, dataset)

    return list(dataset), tokenizer


def loadTokenzier(dataset, file='tokenizer.json'):
    if not os.path.exists(file):
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
            vocab_size=10000,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)
        tokenizer.save(file)
    else:
        tokenizer = Tokenizer.from_file(file)

    return tokenizer


def update_lr(optimizer, step, args):
    warm_step, dim = args.warm_step, args.dim
    lr = (dim**(-0.5))*min(step**(-0.5), step*warm_step**(-1.5))
    for group in optimizer.param_groups:
        group['lr'] = lr


def collate_fn(batch):
    max_cs_len, max_en_len = 0, 0
    for cs, en in batch:
        max_cs_len = max(max_cs_len, len(cs.tokens))
        max_en_len = max(max_en_len, len(en.tokens))
    cs_batch, en_batch = [], []
    cs_mask, en_mask = [], []
    for cs, en in batch:
        mask_idx = len(en.tokens)-1
        cs.pad(max_cs_len, pad_token='[MASK]')
        en.pad(max_en_len, pad_token='[MASK]')
        cs_batch.append(cs.ids)
        cs_mask.append(cs.attention_mask)
        en_batch.append(en.ids)
        en_attention_mask = list(en.attention_mask)
        en_attention_mask[mask_idx] = 0
        en_mask.append(en_attention_mask)
    return cs_batch, en_batch, cs_mask, en_mask


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset(
        'wmt16', 'cs-en', split='train').to_dict()['translation']
    train_data, tokenizer = preprocess(dataset)
    args.vocab_dim = tokenizer.get_vocab_size()

    # train_data = MyDataSet(dataset)
    train_data = DataLoader(train_data, batch_size=args.batch_size,
                            num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = Transformer(args.vocab_dim, args.dim,
                        args.atten_dim, recycle=7).to(device)
    print(f'args:{args}')
    optimizer = torch.optim.Adam(
        model.parameters(), betas=[0.9, 0.98], eps=1e-9)

    model.train()
    for iter in range(args.epoch):
        total_loss = 0
        total_acc = 0
        total_n = 0
        for cs, en, cs_mask, en_mask in tqdm(train_data):
            cs, en = torch.tensor(cs).to(device), torch.tensor(en).to(device)
            cs_mask, en_mask = torch.tensor(cs_mask).to(
                device).detach(), torch.tensor(en_mask).to(device).detach()
            label = en.clone().detach()
            pred = model(cs, en, cs_mask, en_mask)
            loss, acc, n = model.compute_loss(pred, label, en_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_lr(optimizer, iter+1, args)

            total_loss += loss.item()
            total_acc += acc
            total_n += n
        print(
            f'iter:{iter} loss:{total_loss/total_n:.2f} acc:{total_acc/total_n:.2f}')

    validation = load_dataset(
        'wmt16', 'cs-en', split="validation").to_dict()['translation']
    validation_data, _ = preprocess(
        validation, prefix='validation-', tokenizer=tokenizer)

    model.eval()
    for cs, en, cs_mask, en_mask in tqdm(validation_data):
        cs, en = torch.tensor(cs).to(device), torch.tensor(en).to(device)
        cs_mask, en_mask = torch.tensor(cs_mask).to(
            device).detach(), torch.tensor(en_mask).to(device).detach()
        label = en.clone().detach()
        pred = model(cs, en, cs_mask, en_mask)

        non_pad_mask = label.ne(0)
        words = en_mask.nonzero().sum()+1
        p = torch.argmax(pred.view(-1, pred.size(-1)),
                         dim=-1).masked_fill_(non_pad_mask.view(-1), -1)
        assert p.shape == label.shape
        acc = (p == label.view(-1)).sum()
        total_n += words
        total_acc += acc
    print(f'acc:{total_acc/total_n:.2f}')

    torch.save(model, args.save_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dim', type=int, default=512)
    parse.add_argument('--atten_dim', type=int, default=64)
    parse.add_argument('--epoch', type=int, default=1000)
    parse.add_argument('--warm_step', type=int, default=4000)
    parse.add_argument('--max_len', type=int, default=50)
    parse.add_argument('--batch_size', type=int, default=256)
    parse.add_argument('--seed', type=int)
    parse.add_argument('--save_path', type=str,
                       default='model.pt', help='specify path to save model')
    args = parse.parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    main(args)
    print('done')
