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
# print('inputs:')
# print(xb.shape)
print('targets:')
print(yb.shape)
