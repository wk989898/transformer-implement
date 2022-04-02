import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def compute_loss(pred: torch.Tensor, label: torch.Tensor, pad_idx=0, smoothing=0.0):
    non_pad_mask = label.ne(pad_idx)
    p = torch.argmax(pred, dim=-1)
    gt = label
    assert p.shape == gt.shape == non_pad_mask.shape, f'pred shape:{p.shape} and gt shape:{gt.shape} and non_pad_mask shape:{non_pad_mask.shape}'
    acc = p.eq(gt).masked_select(non_pad_mask).sum()
    pred = pred.contiguous().view(-1, pred.size(-1))
    label = label.contiguous().view(-1)
    # Specifies a target value that is ignored and does not contribute to the input gradient.
    loss = F.cross_entropy(
        pred, label, ignore_index=pad_idx, reduction='sum', label_smoothing=smoothing)

    return loss, acc


class Transformer(nn.Module):
    '''
    Rethinking Positional Encoding in Language Pre-training
    https://arxiv.org/abs/2006.15595
    '''

    def __init__(self, vocab_dim, dim, atten_dim, pad_idx=0, pos_len=30, recycle=1, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, dim, padding_idx=pad_idx)
        self.PE = PositionalEncoding(dim, pos_len+50)

        self.encoder = Encoder(
            dim, atten_dim, recycle=recycle, dropout_rate=dropout_rate)
        self.decoder = Decoder(
            dim, atten_dim, recycle=recycle, dropout_rate=dropout_rate)
        self.fc = nn.Linear(dim, vocab_dim)
        self.fc.weight = self.embedding.weight
        self.drop = nn.Dropout(dropout_rate)
        self.dim = dim
        self.pad_idx = pad_idx
        self.pos_len = pos_len

    def generate_mask(self, inputs, outputs):
        out_len = outputs.shape[-1]
        input_mask = inputs.ne(self.pad_idx).unsqueeze(-2)
        subsequent_mask = (
            torch.tril(torch.ones((1, out_len, out_len), device=outputs.device))).bool()
        output_mask = outputs.ne(self.pad_idx).unsqueeze(-2) & subsequent_mask
        return input_mask, output_mask, subsequent_mask

    def embed(self, x):
        # multiply weights by sqrt(dim,-0.5)
        x = self.embedding(x)*self.dim**0.5
        pe = self.PE(x)
        x = self.drop(x)
        return x, pe

    def forward(self, inputs, outputs):
        input_mask, output_mask, subsequent_mask = self.generate_mask(
            inputs, outputs)
        encode, pe_encode = self.embed(inputs)
        encode = self.encoder(encode, pe_encode, input_mask)
        decode, pe_decode, = self.embed(outputs)
        decode = self.decoder(encode, decode, pe_encode,
                              pe_decode, input_mask, output_mask)
        out = self.fc(decode)
        return out

    def translate(self, inputs, outputs, eos_id, beam_size=4):
        '''
        translate one sentence
        '''
        alpha, champion = 0.7, 0
        scores = torch.zeros((beam_size), device=inputs.device)

        input_mask, _, _ = self.generate_mask(inputs, outputs)
        encode, pe_encode = self.embed(inputs)
        encode = self.encoder(encode, pe_encode, input_mask)

        def subsequent_mask(out_len):
            return (torch.tril(torch.ones((1, out_len, out_len), device=outputs.device))).bool()
        # set the maximum output length during inference to input length + 50
        for i in range(self.pos_len+49):
            decode, pe_decode, = self.embed(outputs)
            decode = self.decoder(encode, decode, pe_encode, pe_decode,
                                  input_mask, subsequent_mask(outputs.size(-1)))
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
            if (eos_mask.sum(-1) > 0).sum().item() == beam_size:
                eos_idx = eos_mask.float().argmax(dim=-1)
                # no coverage penalty
                _, champion = (scores/((5+eos_idx)/6)**alpha).max(0)
                break
        return outputs, outputs[champion]


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**(-0.5)

    def forward(self, q, k, v, pe_q, pe_k, mask=None):
        attn = (torch.matmul(q, k.transpose(3, 2)) +
                torch.matmul(pe_q, pe_k.transpose(3, 2)))*self.scale
        if mask is not None:
            mask = rearrange(mask, 'm l1 l2-> m () l1 l2')
            assert mask.shape[-1] == attn.shape[-1], f'masked_fill same size mask:{mask.shape} attention:{attn.shape}'
            attn.masked_fill_(mask == False, -float('inf'))
        attn = torch.softmax(attn, -1)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, atten_dim, head=8, dropout_rate=0.1):
        super().__init__()
        self.head = head
        self.atten = Attention(atten_dim)
        self.q_trans = nn.Linear(dim, head*atten_dim)
        self.k_trans = nn.Linear(dim, head*atten_dim)
        self.v_trans = nn.Linear(dim, head*atten_dim)

        self.pe_q_trans = nn.Linear(dim, head*atten_dim)
        self.pe_k_trans = nn.Linear(dim, head*atten_dim)

        self.fc = nn.Linear(head*atten_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, pe_q, pe_k, mask=None):
        '''
        batch(m) x len(l) x { head(n) x atten_dim(d) }
         return m x l x dim
        '''
        Q = self.q_trans(q)
        K = self.k_trans(k)
        V = self.v_trans(v)
        Q = rearrange(Q, 'm l (n d) -> m n l d', n=self.head)
        K = rearrange(K, 'm l (n d) -> m n l d', n=self.head)
        V = rearrange(V, 'm l (n d) -> m n l d', n=self.head)

        PE_Q = self.pe_q_trans(pe_q)
        PE_Q = rearrange(PE_Q, 'm l (n d) -> m n l d', n=self.head)
        PE_K = self.pe_k_trans(pe_k)
        PE_K = rearrange(PE_K, 'm l (n d) -> m n l d', n=self.head)

        atten = self.atten(Q, K, V, PE_Q, PE_K, mask)
        atten = rearrange(atten, 'm n l d -> m l (d n)')
        atten = self.fc(atten)

        atten = self.drop(atten)
        atten += q
        atten = self.norm(atten)
        return atten


class FeedForward(nn.Module):
    def __init__(self, dim=512, hide_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, hide_dim)
        self.fc2 = nn.Linear(hide_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.drop(x)
        x += residual
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, pos_len=100):
        super().__init__()

        def positional(pos, i):
            return pos/10000**((i//2)/dim)

        PE = [[positional(pos, i) for i in range(dim)]
              for pos in range(pos_len)]
        PE = torch.tensor([PE])
        PE[..., ::2] = torch.sin(PE[..., ::2])
        PE[..., 1::2] = torch.cos(PE[..., 1::2])
        self.register_buffer('PE', PE)

    def forward(self, x):
        return self.PE[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, dim, atten_dim, dropout_rate=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(
            dim, atten_dim, dropout_rate=dropout_rate)
        self.ff = FeedForward(dim, atten_dim, dropout_rate=dropout_rate)

    def forward(self, encode, pe_encode, input_mask):
        encode = self.MHA(encode, encode, encode,
                          pe_encode, pe_encode, input_mask)
        encode = self.ff(encode)
        return encode


class Encoder(nn.Module):
    def __init__(self,  dim, atten_dim, recycle=1, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.ModuleList(
            [EncoderLayer(dim, atten_dim, dropout_rate=dropout_rate) for _ in range(recycle)])

    def forward(self, encode, pe_encode, input_mask=None):
        for layer in self.encoder:
            encode = layer(encode, pe_encode, input_mask)
        return encode


class DecoderLayer(nn.Module):
    def __init__(self, dim, atten_dim, dropout_rate=0.1):
        super().__init__()
        self.MHA1 = MultiHeadAttention(
            dim, atten_dim, dropout_rate=dropout_rate)
        self.MHA2 = MultiHeadAttention(
            dim, atten_dim, dropout_rate=dropout_rate)
        self.ff = FeedForward(dim, atten_dim, dropout_rate=dropout_rate)

    def forward(self, encode, decode, pe_encode, pe_decode, input_mask, output_mask):
        decode = self.MHA1(decode, decode, decode,
                           pe_decode, pe_decode, output_mask)
        decode = self.MHA2(decode, encode, encode,
                           pe_decode, pe_encode, input_mask)
        decode = self.ff(decode)
        return decode


class Decoder(nn.Module):

    def __init__(self,  dim, atten_dim, recycle=1, dropout_rate=0.1):
        super().__init__()
        self.decoder = nn.ModuleList(
            [DecoderLayer(dim, atten_dim, dropout_rate=dropout_rate) for _ in range(recycle)])

    def forward(self, encode, decode, pe_encode, pe_decode, input_mask, output_mask=None):
        for layer in self.decoder:
            decode = layer(encode, decode, pe_encode,
                           pe_decode, input_mask, output_mask)
        return decode
