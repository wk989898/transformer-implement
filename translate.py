import torch
import os
from tokenizers import Tokenizer
from transformer import Transformer

def main(src='',model_path = 'model.pt',file='tokenizer.json'):
    if not os.path.exists(model_path):
        raise FileNotFoundError('model not found')
    state = torch.load(model_path)
    model_dict,args=state['model'],state['args']
    model=Transformer(args.vocab_dim, args.dim, args.atten_dim,
                        pad_idx=args.pad_idx, pos_len=args.max_len, recycle=6).cuda()
    model.load_state_dict(model_dict)
    tokenizer =Tokenizer.from_file(file)

    model.eval()
    with torch.no_grad():
        src, target = tokenizer.encode(src), [tokenizer.token_to_id('<BOS>')]
        src = torch.tensor([src.ids]).cuda()
        target = torch.tensor([target]).cuda()
        result, target = model.translate(
            src, target, eos_id=tokenizer.token_to_id('<EOS>'), beam_size=2)

        print(f'target: {tokenizer.decode(target.tolist())}')
        print(f'result: {tokenizer.decode_batch(result.tolist())}')


if __name__ == '__main__':
    src = 'attention is all you need'
    main(src)
