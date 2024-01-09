import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import torch.nn.functional as F
#from dataset import *
import math
import config
from utils import make_patches, calc_num_patches
from dataset import FlickerDS, MyTokenizer, get_img2discr

#
# I copied 'PositionalEncoding' it from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 'get_tgt_mask' i copid from here: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def get_tgt_mask(size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

#iage caption transformer    
class PrintM(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
    
class ICTrans(nn.Module):
    def __init__(self, n_patches, embedding_size, num_heads, num_layers, vocab_size, dropout=0.1):
        super().__init__()

        # x(image patch) to embedding + positional encodeing
        input_embed_size = (config.PATCH_SIZE*n_patches)**2
        self.img_embed = nn.Linear(3*(config.PATCH_SIZE**2), embedding_size)
        self.patch_xpos_embed = nn.Embedding(n_patches**2, embedding_size)
        # i wanna tre pos encode image use x and y then i just 
        #self.patch_ypos_embed = nn.Embedding(n_patches, embedding_size)

        # y(tokenized sequence) to embeding + positional encodeing
        self.vocab_embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_enc = PositionalEncoding(d_model=embedding_size, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        #raise RuntimeError('It must return probs for every character')
        self.lin = nn.Sequential(
            nn.Linear(embedding_size, vocab_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(vocab_size*2, vocab_size))
    
    def forward(self, x, y):
        y = self.vocab_embed(y)
        y = y.permute(1, 0, 2) #from (bs, seq_len, features) to (seq_len, bs, features)
        y += self.pos_enc(y)
        # from (bs, ch, xpn, ypn, xps, yps) to (xpn, ypn, bs, ch, xps, yps)
        # bs - batch_size, ch - chanels, xpn - n patches on x axis, xps - x patch size
        x = x.permute(2, 3, 0, 1, 4, 5)

        # from (xpn, ypn, bs, ch, xps, yps) to (xpn *ypn, bs, ch*xps*yps ) = (sequence_length, batch_size, features) 
        xpn, ypn, bs, ch, xps, yps = x.shape
        x = x.reshape(xpn, ypn, bs, ch*xps*yps).view(xpn * ypn, bs, ch*xps*yps) 
        #print('sh after all', x.shape)
        pos = torch.arange(0, xpn**2, dtype=torch.long, device=config.DEVICE)
        
        #print('x', x.shape) 
        #print('x img emb', self.img_embed(x).shape)
        x = self.img_embed(x) + self.patch_xpos_embed(pos).unsqueeze(dim=1)# + self.patch_ypos_embed(pos)
        tgt_mask = get_tgt_mask(y.shape[0])

        t_out = self.transformer(src= x, tgt=y, tgt_mask = tgt_mask)
        probs = self.lin(t_out.permute(1, 0, 2)) #permute: (seq_len, bs, emb_size) -> (bs, seq_len, emb_size))
        return probs
    
    

def dim_test():
    #x = torch.randn(8, 8, 32, 3, 64, 64)
    npthcs = calc_num_patches()
    x = torch.randn(npthcs, npthcs, config.BATCH_SIZE, 3, config.PATCH_SIZE, config.PATCH_SIZE)
    print('x test', x.shape)
    xpn, ypn, bs, ch, xps, yps = x.shape
    x = x.view(xpn, ypn, bs, ch*xps*yps).view(xpn * ypn, bs, ch*xps*yps)
    print(x.shape)

def ict_test():
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())
    
    ds = FlickerDS(img_folder_path=config.IMG_FOLDER_PATH,
                   img2descr=img2descr_lemma,
                   img_names=config.TEST_IMG_NAMES,
                   img_size = config.IMG_SIZE,
                   tokenizer=tokenizer)
    dl = DataLoader(ds,
                    batch_size= config.BATCH_SIZE, 
                    shuffle=True, 
                    #num_workers=config.NUM_WORKERS,
                    pin_memory=True)
    
    model = ICTrans(n_patches= calc_num_patches(),
                    embedding_size = config.EMBED_SIZE,
                    num_heads = config.N_HEADS,
                    num_layers = config.N_TRANS_LAYERS,
                    vocab_size = vocab_size)
    
    for img_batch, discr_batch in dl:
        print('before patch devision',img_batch.shape, discr_batch.shape)
        
        img_batch = make_patches(img_batch, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
        print('after patch division',img_batch.shape, discr_batch.shape)
        
        out = model(img_batch, discr_batch)
        print('model out', out.shape)

        words = tokenizer.probs2words(out[0])
        print(words)
        #print(f'probs for one letter {out[0][0][1]} and it must be {1/vocab_size}') # iput must be gaussian distr #if not output is not uniform distribution
        break
if __name__ == '__main__': 
    print('#'*50)
    print('#'*50)
    print('#'*50)
    print('#'*50)

    #dim_test()
    ict_test()
