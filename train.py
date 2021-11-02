import torch 
from transformer import Transformer
from torch.utils.data import DataLoader


train_data=DataLoader(data_set,batch_size=4,num_workers=4,shuffle=True)

dim=64
atten_dim=128
vocab_dim=1024

model=Transformer(vocab_dim, dim, atten_dim)
optimizer=torch.optim.Adam(model.parameters(),betas=[0.9,0.98],eps=1e-9)

for (inputs,outputs),labels in train_data:
    pred=model(inputs,outputs)
    loss=model.loss(pred,labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()