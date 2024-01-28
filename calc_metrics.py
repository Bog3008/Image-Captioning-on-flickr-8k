import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
#import torch.nn.functional as F
#from dataset import *
import math
import os
from datetime import datetime
from tqdm import tqdm

import config
import utils
from utils import make_patches, calc_num_patches
from dataset import FlickerDS, MyTokenizer, get_img2discr
from model import ICTrans, RTnet, CTnet
from train import get_train_test_dl, evaluate, evaluate_iference

def iference_bleu1234(model, dloader, tokenizer):
    model.eval()
    avg_bleu = torch.tensor([0.0, 0.0, 0.0, 0.0])
    i_batches_passed=1
    #padd_tensor = torch.full((config.BATCH_SIZE, 1), tokenizer.pad_idx).to(config.DEVICE)

    for i ,(img_batch, descr_batch) in enumerate(tqdm(dloader, leave=False)):
        img_batch=img_batch.to(config.DEVICE)
        descr_batch=descr_batch.to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            
            tokens = model.inference(img_batch)
            for bleu_type in [1, 2, 3, 4]:
                avg_bleu[bleu_type-1] += utils.calc_bleu(tokens, descr_batch, tokenizer, bleu_type)
            #print(aaa, type(aaa))
            #avg_bleu += utils.calc_bleu(tokens, descr_batch, tokenizer, bleu_type)
    
    return avg_bleu/len(dloader)

def calc_param_num(model):
    params = list(model.parameters())
    param_count = sum([p.numel() for p in params])
    return param_count

def run(model_type='ICT'):
    #vocab and tokenizer
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())

    #dataloaders
    train_dl, test_dl = get_train_test_dl(img2descr_lemma, tokenizer, shuffle_test=False, shuffle_train=False)
    
    #model & optim
    scaler = torch.cuda.amp.GradScaler()

    model_params = {"n_patches": calc_num_patches(),
               "embedding_size": config.EMBED_SIZE,
               "num_heads": config.N_HEADS,
               "num_layers": config.N_TRANS_LAYERS,
               "vocab_size": vocab_size,
               "bos_idx": tokenizer.bos_idx,
               "eos_idx": tokenizer.eos_idx,
               "dropout": config.DROPOUT}
    if model_type == 'ICT':
        ict_model = ICTrans(**model_params)
    if model_type == 'RT':
        ict_model = RTnet(**model_params)
    if model_type == 'CT':
        ict_model = CTnet(**model_params)

    ict_model = ict_model.to(config.DEVICE)
    optimizer = config.OPTIMIZER(ict_model.parameters(), lr = config.LR)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, size_average=True)
    
    if config.LOAD_MODEL:
        print('loading model...')
        utils.load_model(ict_model, optimizer, config.LOAD_MODEL_NAME)
    
    #utils.img_and_descr(ict_model, train_dl, tokenizer, n_imgs=3)
    #utils.img_and_descr(ict_model, test_dl, tokenizer, n_imgs=3)
    #return
    os.system('cls')
    print("Total model parameters:",calc_param_num(ict_model))
    for dl, name in [(train_dl, 'Train'), (test_dl, 'Test')]:
        print(name, 'dataset:')
        utils.show_descr(model=ict_model, dl=dl, tokenizer=tokenizer, title='train sentence comparison')
        #for bleu_type in [1, 2, 3, 4]:
        infer_bleu = iference_bleu1234( model=ict_model, dloader=dl, tokenizer=tokenizer)
        for i in range(1, 5):
            print(f'bleu_{i}:', f'{float(infer_bleu[i-1]):.4f}', end='; ')
        

def show_examples(ict_model, train_dl, tokenizer):
    utils.img_and_descr(ict_model, train_dl, tokenizer, n_imgs=4)
    
if __name__ == '__main__':
    print('}{'*20)
    run()