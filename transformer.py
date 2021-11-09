from operator import le
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def compute_loss(pred: torch.Tensor, label: torch.Tensor, pad_idx=0, smoothing=False, vocab_dim=10000):
    non_pad_mask = label.ne(pad_idx)

    p = torch.argmax(pred, dim=-1)
    gt = label
    assert p.shape == gt.shape, f'pred shape:{p.shape} and gt shape:{gt.shape}'
    acc = p.eq(gt).masked_select(non_pad_mask).sum()

    if smoothing:
        label = F.one_hot(label, num_classes=vocab_dim)
        eps = 0.1
        label = label * (1 - eps) + (1 - label) * \
            eps / (vocab_dim - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(label * log_prb).sum(dim=-1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        pred = pred.view(-1, pred.size(-1))
        label = label.view(-1)
        # Specifies a target value that is ignored and does not contribute to the input gradient.
        loss = F.cross_entropy(pred, label, ignore_index=pad_idx,reduction='sum')

    return loss, acc


class Transformer(nn.Module):
    def __init__(self, vocab_dim, dim, atten_dim, pad_idx=0, recycle=1):
        super().__init__()
        self.encoder = Encoder(vocab_dim, dim, atten_dim,
                               pad_idx=pad_idx, recycle=recycle)
        self.decoder = Decoder(vocab_dim, dim, atten_dim,
                               pad_idx=pad_idx, recycle=recycle)
        self.fc = nn.Linear(dim, vocab_dim)

        self.pad_idx = pad_idx

    def generate_mask(self, inputs, outputs):
        input_mask = inputs.ne(self.pad_idx).unsqueeze(-2).clone()
        output_mask = outputs.ne(self.pad_idx).unsqueeze(-2).clone()
        out_len = outputs.shape[-1]
        output_mask = output_mask & (
            1-torch.triu(torch.ones((1, out_len, out_len), device=outputs.device), diagonal=1)).bool()
        return input_mask, output_mask

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor):
        input_mask, output_mask = self.generate_mask(inputs, outputs)

        encode = self.encoder(inputs, input_mask)
        decode = self.decoder(encode, outputs, input_mask, output_mask)

        out = torch.softmax(self.fc(decode), dim=-1)
        return out


class Attention(nn.Module):
    def __init__(self, dim, pad_idx):
        super().__init__()
        self.tem = dim**0.5
        self.pad_idx = pad_idx

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        '''
        parallel
         m x n x l x d
        '''
        attn = torch.matmul(q/self.tem, k.transpose(3, 2))
        if mask is not None:
            mask = rearrange(mask, 'm l1 l2-> m () l1 l2')
            assert mask.shape[-1] == attn.shape[-1], f'masked_fill same size mask:{mask.shape} attention:{attn.shape}'
            attn.masked_fill_(mask == self.pad_idx, -1e9)
        attn = torch.softmax(attn, -1)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, atten_dim, pad_idx=0, head=8):
        super().__init__()
        self.head = head
        self.atten = Attention(atten_dim, pad_idx)
        self.q_trans = nn.Linear(dim, head*atten_dim)
        self.k_trans = nn.Linear(dim, head*atten_dim)
        self.v_trans = nn.Linear(dim, head*atten_dim)
        self.fc = nn.Linear(head*atten_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.drop = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
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
        atten = self.atten(Q, K, V, mask)
        atten = rearrange(atten, 'm n l d -> m l (d n)')
        atten = self.fc(atten)

        atten = self.drop(atten)
        atten += q
        atten = self.norm(atten)
        return atten


class FeedForward(nn.Module):
    def __init__(self, dim=512, hide_dim=2048):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, hide_dim)
        self.fc2 = nn.Linear(hide_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        out += x
        return self.norm(out)


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
        return self.PE[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, vocab_dim, dim, atten_dim, pad_idx, recycle=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_dim, dim, padding_idx=pad_idx)
        self.MHA = MultiHeadAttention(dim, atten_dim, pad_idx=pad_idx)
        self.PE = PositionalEncoding(dim)
        self.ff = FeedForward(dim)
        self.recycle = recycle

    def forward(self, words: torch.LongTensor, mask=None):
        encode = self.embed(words)
        encode += self.PE(encode)
        # recycle n times
        for _ in range(self.recycle):
            encode = self.MHA(encode, encode, encode, mask)  # self-attention
            encode = self.ff(encode)

        return encode


class Decoder(nn.Module):

    def __init__(self, vocab_dim, dim, atten_dim, pad_idx, recycle=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_dim, dim, padding_idx=pad_idx)
        self.MHA = MultiHeadAttention(dim, atten_dim, pad_idx=pad_idx)
        self.PE = PositionalEncoding(dim)
        self.ff = FeedForward(dim)
        self.recycle = recycle

    def forward(self, encode, words: torch.LongTensor, input_mask, mask=None):
        decode = self.embed(words)
        decode += self.PE(decode)
        # recycle n times
        for _ in range(self.recycle):
            decode = self.MHA(decode, decode, decode, mask)  # self-attention
            decode = self.MHA(decode, encode, encode, input_mask)
            decode = self.ff(decode)

        return decode
