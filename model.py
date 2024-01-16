import torch
import torch.nn as nn
import torchvision.models as models
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
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 
        return mask

#iage caption transformer    
class PrintM(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
    
class ICTrans(nn.Module):
    def __init__(self, n_patches, embedding_size, num_heads, num_layers, vocab_size, bos_idx, eos_idx, dropout=0.1):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
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
        '''
        x - images batch
        y - tokenized description batch
        '''
        y = self.prepare_y(y)
        x = self.prepare_x(x)
        tgt_mask = get_tgt_mask(y.shape[0])
        t_out = self.transformer(src= x, tgt=y, tgt_mask = tgt_mask)
        probs = self.lin(t_out.permute(1, 0, 2)) #permute: (seq_len, bs, emb_size) -> (bs, seq_len, emb_size))
        return probs
    def prepare_x(self, x):
        x = make_patches(x, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
        # from (bs, ch, xpn, ypn, xps, yps) to (xpn, ypn, bs, ch, xps, yps)
        # bs - batch_size, ch - chanels, xpn - n patches on x axis, xps - x patch size
        x = x.permute(2, 3, 0, 1, 4, 5)
        # from (xpn, ypn, bs, ch, xps, yps) to (xpn *ypn, bs, ch*xps*yps ) = (sequence_length, batch_size, features) 
        xpn, ypn, bs, ch, xps, yps = x.shape
        x = x.reshape(xpn, ypn, bs, ch*xps*yps).view(xpn * ypn, bs, ch*xps*yps) 
        pos = torch.arange(0, xpn**2, dtype=torch.long, device=config.DEVICE)
        x = self.img_embed(x) + self.patch_xpos_embed(pos).unsqueeze(dim=1)# + self.patch_ypos_embed(pos)
        return x
    def prepare_y(self, y):
        y = self.vocab_embed(y)
        y = y.permute(1, 0, 2) #from (bs, seq_len, features) to (seq_len, bs, features)
        y += self.pos_enc(y)
        return y
    
    def inference(self, x):
        '''
        x - image batch
        '''
        x = self.prepare_x(x)
        bs = x.shape[1] 
        enc_out = self.transformer.encoder(x)
        
        tokens_in = torch.full((bs, 1), self.bos_idx, dtype=torch.long)
        tokens_in = tokens_in.to(config.DEVICE)

        for i in range(config.MAX_SEQ_LEN + 2): # +2 for bos and eos
            dec_in = self.prepare_y(tokens_in)
            dec_out = self.transformer.decoder(tgt=dec_in, memory=enc_out)
            probs = self.lin(dec_out.permute(1, 0, 2))
            tokens = torch.argmax(probs, dim=2)
            tokens = tokens[:, -1].unsqueeze(1)
            #print('token shape', tokens.shape)
            #print('tokens_in shape', tokens_in.shape)
            tokens_in = torch.concat((tokens_in, tokens), dim=1)
            #ADD only last token
            #print(tokens_in)
        return tokens_in


class RTnet(nn.Module):
    def __init__(self, n_patches, embedding_size, num_heads, num_layers, vocab_size, bos_idx, eos_idx, dropout=0.1):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        
        # y(tokenized sequence) to embeding + positional encodeing
        self.vocab_embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_enc = PositionalEncoding(d_model=embedding_size, dropout=dropout)

        self.backbone, out_shape = self.get_backbone()
        # out_shape may look like 1x512x7x7. 512 too much i want 16
        bs, n_features, x_size, y_size = out_shape
        self.img_compres = nn.Conv2d(n_features, 16, kernel_size=(1, 1))
        self.img_embed = nn.Linear(x_size*y_size, embedding_size)

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
        '''
        x - images batch
        y - tokenized description batch
        '''
        y = self.prepare_y(y)
        x = self.prepare_x(x)
        tgt_mask = get_tgt_mask(y.shape[0])
        t_out = self.transformer(src= x, tgt=y, tgt_mask = tgt_mask)
        probs = self.lin(t_out.permute(1, 0, 2)) #permute: (seq_len, bs, emb_size) -> (bs, seq_len, emb_size))
        return probs
    
    def prepare_x(self, x):
        x = self.backbone(x) # 1x512x7x7
        x = self.img_compres(x)
        bs, n_p, x_size, y_size = x.shape
        #print('COMPRESSION', x.shape)
        #print('COMPRESSION view', x.view(bs, n_p, x_size * y_size).shape)
        x = x.view(n_p, bs, x_size * y_size)
        x = self.img_embed(x)
        return x
    def prepare_y(self, y):
        y = self.vocab_embed(y)
        y = y.permute(1, 0, 2) #from (bs, seq_len, features) to (seq_len, bs, features)
        y += self.pos_enc(y)
        return y
    
    def get_backbone(self):
        resnet18 = models.resnet18(pretrained=True)
        newmodel = torch.nn.Sequential(*(list(resnet18.children())[:-2])) # kick adaptive avgpool and linear layer
        for param in newmodel.parameters():
            param.requires_grad = False
        return newmodel, torch.Size([1, 512, 7, 7]) # out is 1x512x7x7
    

    def inference(self, x):
        '''
        x - image batch
        '''
        x = self.prepare_x(x)
        bs = x.shape[1] 
        enc_out = self.transformer.encoder(x)
        
        tokens_in = torch.full((bs, 1), self.bos_idx, dtype=torch.long)
        tokens_in = tokens_in.to(config.DEVICE)

        for i in range(config.MAX_SEQ_LEN + 2): # +2 for bos and eos
            dec_in = self.prepare_y(tokens_in)
            dec_out = self.transformer.decoder(tgt=dec_in, memory=enc_out)
            probs = self.lin(dec_out.permute(1, 0, 2))
            tokens = torch.argmax(probs, dim=2)
            tokens = tokens[:, -1].unsqueeze(1)
            #print('token shape', tokens.shape)
            #print('tokens_in shape', tokens_in.shape)
            tokens_in = torch.concat((tokens_in, tokens), dim=1)
            #ADD only last token
            #print(tokens_in)
        return tokens_in
    
class CTnet(nn.Module):
    def __init__(self, n_patches, embedding_size, num_heads, num_layers, vocab_size, bos_idx, eos_idx, dropout=0.1):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # y(tokenized sequence) to embeding + positional encodeing
        self.vocab_embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_enc = PositionalEncoding(d_model=embedding_size, dropout=dropout)

        #encoder
        self.backbone, out_shape = self.get_backbone()
        # out_shape may look like 1x512x7x7. 512 too much i want 16
        bs, n_features, x_size, y_size = out_shape
        self.img_compres = nn.Conv2d(n_features, 32, kernel_size=(1, 1))
        self.img_embed = nn.Linear(x_size*y_size, embedding_size)
        
        #decoder
        full_transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = full_transformer.decoder # why dont use TransformerDecoder?
        #I took a look at https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder
        # get decoder from transormer easier than define TransformerDecoderLayer, TransformerDecoder and LayerNorm separately
        
        self.lin = nn.Sequential(
            nn.Linear(embedding_size, vocab_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(vocab_size*2, vocab_size))
    
    def forward(self, x, y):
        '''
        x - images batch
        y - tokenized description batch
        '''
        y = self.prepare_y(y)
        tgt_mask = get_tgt_mask(y.shape[0])
        encoded_x = self.prepare_x(x)

        encoder_out = self.decoder(tgt=y, memory=encoded_x, tgt_mask=tgt_mask)

        probs = self.lin(encoder_out.permute(1, 0, 2)) #permute: (seq_len, bs, emb_size) -> (bs, seq_len, emb_size))
        return probs
    
    def prepare_x(self, x):
        x = self.backbone(x) # smth like 1x512x7x7
        x = self.img_compres(x) # smth like 1x16x7x7
        bs, n_p, x_size, y_size = x.shape
        x = x.view(n_p, bs, x_size * y_size) # smth like 16x1x49
        x = self.img_embed(x) # smth like 16x1x256 # if embed size=256 
        return x
    
    def prepare_y(self, y):
        y = self.vocab_embed(y)
        y = y.permute(1, 0, 2) #from (bs, seq_len, features) to (seq_len, bs, features)
        y += self.pos_enc(y)
        return y
    
    def get_backbone(self):
        resnet18 = models.resnet18(pretrained=True)
        newmodel = torch.nn.Sequential(*(list(resnet18.children())[:-2])) # kick adaptive avgpool and linear layer
        for param in newmodel.parameters():
            param.requires_grad = False
        return newmodel, torch.Size([1, 512, 7, 7]) # out is 1x512x7x7
    

    def inference(self, x):
        '''
        x - image batch
        '''
        encoder_out = self.prepare_x(x)
        bs = encoder_out.shape[1] 
        
        tokens_in = torch.full((bs, 1), self.bos_idx, dtype=torch.long)
        tokens_in = tokens_in.to(config.DEVICE)

        for i in range(config.MAX_SEQ_LEN + 2): # +2 for bos and eos
            dec_in = self.prepare_y(tokens_in)
            dec_out = self.decoder(tgt=dec_in, memory=encoder_out)
            probs = self.lin(dec_out.permute(1, 0, 2))

            tokens = torch.argmax(probs, dim=2)
            tokens = tokens[:, -1].unsqueeze(1)
            #print('token shape', tokens.shape)
            #print('tokens_in shape', tokens_in.shape)
            tokens_in = torch.concat((tokens_in, tokens), dim=1)
            #ADD only last token
            #print(tokens_in)
        return tokens_in

def dim_test():
    #x = torch.randn(8, 8, 32, 3, 64, 64)
    npthcs = calc_num_patches()
    x = torch.randn(npthcs, npthcs, config.BATCH_SIZE, 3, config.PATCH_SIZE, config.PATCH_SIZE)
    print('x test', x.shape)
    xpn, ypn, bs, ch, xps, yps = x.shape
    x = x.view(xpn, ypn, bs, ch*xps*yps).view(xpn * ypn, bs, ch*xps*yps)
    print(x.shape)

def ict_test(model_name='ICT'):
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
    
    model_params = {
    "n_patches": calc_num_patches(),
    "embedding_size": config.EMBED_SIZE,
    "vocab_size": vocab_size,
    "bos_idx": tokenizer.bos_idx,
    "eos_idx": tokenizer.eos_idx,
    "num_heads": config.N_HEADS,
    "num_layers": config.N_TRANS_LAYERS,
    }
    
    if model_name == 'ICT':
        model = ICTrans(**model_params)
    if model_name == 'RT':
        model = RTnet(**model_params)
    if model_name == 'CT':
        model = CTnet(**model_params)
    

    model.to(config.DEVICE)
    for img_batch, discr_batch in dl:
        img_batch, discr_batch = img_batch.to(config.DEVICE), discr_batch.to(config.DEVICE)
        print('before patch devision',img_batch.shape, discr_batch.shape)
        
        out = model(img_batch, discr_batch)
        print('model out', out.shape)

        words = tokenizer.probs2words(out[0])
        print(' '.join(words))
        #print(f'probs for one letter {out[0][0][1]} and it must be {1/vocab_size}') # iput must be gaussian distr #if not output is not uniform distribution
        break
def itc_inference_test(model_name='ICT'):
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
    
    model_params = {
    "n_patches": calc_num_patches(),
    "embedding_size": config.EMBED_SIZE,
    "vocab_size": vocab_size,
    "bos_idx": tokenizer.bos_idx,
    "eos_idx": tokenizer.eos_idx,
    "num_heads": config.N_HEADS,
    "num_layers": config.N_TRANS_LAYERS,
    }
    
    if model_name == 'ICT':
        model = ICTrans(**model_params)
    if model_name == 'RT':
        model = RTnet(**model_params)
    if model_name == 'CT':
        model = CTnet(**model_params)

    model.to(config.DEVICE)

    for img_batch, discr_batch in dl:
        img_batch, discr_batch = img_batch.to(config.DEVICE), discr_batch.to(config.DEVICE)
        print('before patch devision',img_batch.shape, discr_batch.shape)
        
        out = model.inference(img_batch)
        print('model out', out.shape)

        words_list_list = tokenizer.decode_batch(out)#probs2words(out[0])
        for words_list in words_list_list:
            print(' '.join(words_list))
        #print(f'probs for one letter {out[0][0][1]} and it must be {1/vocab_size}') # iput must be gaussian distr #if not output is not uniform distribution
        return
if __name__ == '__main__': 
    print('#'*50)
    print('#'*50)
    print('#'*50)
    print('#'*50)

    #dim_test()
    #ict_test('CT')
    itc_inference_test('CT')
