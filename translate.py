import torch
import torch.nn.functional as F
from einops import repeat
from tokenizers import Tokenizer
import os


def translate(self, inputs, outputs, eos_id, beam_size=4):
        '''
        translate one sentence
        same as model.translate()
        '''
        alpha, champion = 0.7, 0
        scores = torch.zeros((beam_size), device=inputs.device)

        input_mask, _, _ = self.generate_mask(inputs, outputs)
        encode = self.encoder(self.embed(inputs), input_mask)
        
        def subsequent_mask(out_len):
            return (torch.tril(torch.ones((1, out_len, out_len), device=outputs.device))).bool()
        # set the maximum output length during inference to input length + 50
        for i in range(self.pos_len+49):
            decode = self.decoder(encode, self.embed(
                outputs), input_mask, subsequent_mask(outputs.size(-1)))
            pred = self.fc(decode)
            rank = F.log_softmax(pred[:, -1], dim=-1)
            # search topk: beam_size x vocab_size -> beam_size x beam_size
            current_win, current_token = rank.topk(beam_size)
            scores = scores+current_win
            scores, winners = scores.view(-1).topk(beam_size)
            select_token = torch.index_select(
                current_token.view(-1), 0, winners)
            if i == 0:
                outputs = repeat(outputs, '() b -> beam b', beam=beam_size)
                # encode shape is (batch_size,beam_size,hidden_size)
                encode = repeat(encode, '() b d -> beam b d', beam=beam_size)
            outputs = torch.cat([outputs, select_token.unsqueeze(-1)], dim=-1)

            eos_mask = outputs == eos_id
            # every beam has eos token
            if (eos_mask.sum(-1)>0).sum().item() == beam_size:
                eos_idx = eos_mask.float().argmax(dim=-1)
                # no coverage penalty
                _, champion = (scores/((5+eos_idx)/6)**alpha).max(0)
                break
        return outputs, outputs[champion]


if __name__ == '__main__':
    model_path = 'model.pt'
    if not os.path.exists(model_path):
        raise ValueError('model not found')
    state = torch.load(model_path)
    model, tokenizer = state['model'], Tokenizer.from_str(state['tokenizer'])

    model.eval()
    with torch.no_grad():
        src = ''
        src, en = tokenizer.encode(src), [tokenizer.token_to_id('<BOS>')]
        print(f'src {src.ids} {src.tokens}')
        src = torch.tensor([src.ids]).cuda()
        en = torch.tensor([en]).cuda()
        result, target = model.translate(src, en, eos_id=tokenizer.token_to_id('<EOS>'),beam_size=4)
        # result, target = translate(model, src, en, eos_id=tokenizer.token_to_id('<EOS>'),beam_size=4)

        print(f'target: {tokenizer.decode(target.tolist())}')
        print(f'result: {tokenizer.decode_batch(result.tolist())}')