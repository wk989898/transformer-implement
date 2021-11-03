from transformer import MultiHeadAttention,Transformer,PositionalEncoding
import torch

dim=64
atten_dim=128
vocab_dim=1024

model=MultiHeadAttention(dim,atten_dim)

input_size=(2,10,dim)
q=torch.randn(input_size)
out=model(q,q,q)
assert out.shape==torch.Size(input_size),'same size'
# print(out)


inputs=torch.arange(0,vocab_dim).unsqueeze(0).long()
outputs=torch.arange(0,vocab_dim).unsqueeze(0).long()
model=Transformer(vocab_dim, dim,atten_dim)
pred=model(inputs,outputs)
assert pred.shape==torch.Size((1,vocab_dim,1)),'assert size'
# print(pred)

res=PositionalEncoding(vocab_dim,dim)
assert res.shape==torch.Size([vocab_dim,dim]),'positional'
# print(res)