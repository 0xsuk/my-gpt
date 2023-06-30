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

