with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print([:1000])

# all the unieuq chars in alphabet order
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)  # 65

stoi = { ch:i for i,ch in enumerate(chars) } # string to intergers
itos = { i:ch for i,ch in enumerate(chars) } # integers to string
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # integers to string

# print(decode(encode("hii thererererereeh")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337) # random number generator
batch_size = 4 # the num of sets of data trained simultaneously
block_size = 8 # the num of chars transformer use to predict the next char

def get_batch(split):
    # generate a small batch of data of inputs x and targets y 
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # list of random offsets of data with len of batch_size. subtracting block size to prevent overflow
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack of set of data with len of batch_size. shape = [batch_size(row), block_size(col)]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # answers
    return x, y

xb, yb = get_batch('train')
# print(xb)
# xb = tensor([[24, 43, 58,  5, 57,  1, 46, 43],
#             [44, 53, 56,  1, 58, 46, 39, 58],
#             [52, 58,  1, 58, 46, 39, 58,  1],
#             [25, 17, 27, 10,  0, 21,  1, 54]]) (4, 8)
# print('inputs:')
# print(xb.shape)
# print('targets:')
# print(yb.shape)


import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets):
        # predict scores for the next char
        logits = self.token_embedding_table(idx) # (Batch, Time, Channel(tensor))
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C) # reshape (B,T,C) to (B*T, C) to pass it to cross_entry function
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
