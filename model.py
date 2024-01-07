import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#import torch.nn.functional as F
from dataset import *
import math
import config
from utils import make_patches, calc_num_patches

    
## Model in the flesh(in code actually)
#iage caption transformer    
def ICTrans(nn.Module):
    def __init__(self, n_patches, embedding_size, num_heads, num_layers, num_classes, dropout=0.1):
        input_embed_size = (config.PATCH_SIZE*n_patches)**2
        self.img_embed = nn.Embeding(input_embed_size, embedding_size)
        self.xpos_embed = nn.Embeding(n_patches, embedding_size)
        self.ypos_embed = nn.Embeding(n_patches, embedding_size)

        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
    
    def forward(self, x):
        # from (bs, ch, xpn, ypn, xps, yps) to (xpn, ypn, bs, ch, xps, yps)
        # bs - batch_size, ch - chanels, xpn - n patches on x axis, xps - x patch size
        x = x.permute(2, 3, 0, 1, 4, 5)
        x_patch_num, y_patch_num = x.shape[:2]  
        x = self.img_embed(x) + self.xpos(x_patch_num) + self.ypos(x_patch_num)
