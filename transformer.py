import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Transformer(nn.Module):
    def __init__(self, dim, atten_dim):
        super().__init__()
        self.dim = dim
        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)
        self.MHA = MultiHeadAttention(dim, atten_dim)
        # 
        self.fc = nn.Linear(dim,1) 

    def forward(self, inputs, outputs):
        encode = self.encoder(inputs)
        # recycle 
        decode = self.decoder(outputs)
        out = self.MHA(encode, encode, decode)
        #
        out = torch.softmax(self.fc(out), dim=-1)
        return out


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tem = torch.sqrt(self.dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        '''
        parallel
         n x l x d
        '''
        attn = torch.mul(q, k.transpose(2, 1))/self.tem
        attn = self.ln(attn)
        if mask:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, -1)
        return torch.mul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, atten_dim, head=8):
        super().__init__()
        self.head = head
        self.atten = Attention(atten_dim)
        self.q_trans = nn.Linear(dim, head*atten_dim)
        self.k_trans = nn.Linear(dim, head*atten_dim)
        self.v_trans = nn.Linear(dim, head*atten_dim)
        self.fc = nn.Linear(head*atten_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask=None):
        '''
        len(l) x { head(n) x atten_dim(d) }
         return l x dim
        '''
        Q = self.q_trans(q)
        K = self.k_trans(k)
        V = self.v_trans(v)
        Q = rearrange(Q, 'l (n d) -> n l d', n=self.head)
        K = rearrange(K, 'l (n d) -> n l d', n=self.head)
        V = rearrange(V, 'l (n d) -> n l d', n=self.head)
        atten = self.atten(Q, K, V, mask)
        atten = torch.cat(atten, dim=0)
        print('atten shape', atten.shape)
        atten = self.fc(atten)
        atten += q
        atten = self.norm(atten)
        return atten


class FeedForward(nn.Module):
    def __init__(self, dim=512, hide_dim=2048):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, hide_dim)
        self.fc2 = nn.Linear(hide_dim, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x), inplace=True))


def PositionalEncoding(dim, pos_len):
    def positional(pos, i):
        if i & 1 == 1:
            return torch.cos(pos/10000**((i//2)/dim))
        else:
            return torch.sin(pos/10000**((i//2)/dim))

    PE = torch.zeros((dim, pos_len))
    for pos in range(pos_len):
        for i in range(dim):
            PE[pos][i] = positional(pos, i)
    return PE


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding()
        self.MHA = MultiHeadAttention()

    def forward(self, words:torch.Tensor):
        encode = self.embed(words)
        encode += PositionalEncoding(self.dim, words.shape[0])
        # recycle n times
        encode = self.MHA(encode,encode,encode) # self-attention

        return encode


class Decoder(nn.Module):
    '''
    same as encoder，but no recycling
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding()

    def forward(self, words:torch.Tensor):
        decode = self.embed(words)
        decode += PositionalEncoding(self.dim, words.shape[0])
        decode = self.MHA(decode,decode,decode) # self-attention
        
        return decode
