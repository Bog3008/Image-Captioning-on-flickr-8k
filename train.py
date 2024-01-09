import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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
from model import ICTrans

def get_train_test_dl(img2descr_lemma, tokenizer):
    train_ds = FlickerDS(img_folder_path=config.IMG_FOLDER_PATH,
                   img2descr=img2descr_lemma,
                   img_names=config.TRAIN_IMG_NAMES,
                   img_size = config.IMG_SIZE,
                   tokenizer=tokenizer)
    train_dl = DataLoader(train_ds,
                    batch_size= config.BATCH_SIZE, 
                    shuffle=True, 
                    #num_workers=config.NUM_WORKERS,
                    pin_memory=True)
    
    test_ds = FlickerDS(img_folder_path=config.IMG_FOLDER_PATH,
                   img2descr=img2descr_lemma,
                   img_names=config.TEST_IMG_NAMES,
                   img_size = config.IMG_SIZE,
                   tokenizer=tokenizer)
    test_dl = DataLoader(test_ds,
                    batch_size= config.BATCH_SIZE, 
                    shuffle=True, 
                    #num_workers=config.NUM_WORKERS,
                    pin_memory=True)
    return train_dl, test_dl
def print_elapsed_time(elapsed_time, text='time pre epo'):
    minutes = int(elapsed_time.total_seconds() // 60)
    seconds = int(elapsed_time.total_seconds() % 60)
    os.system('cls')
    print(f'{text} {minutes}m {seconds}s')

def train(model, optimizer, criterion , scaler, dloader, writer=None):
    model.train()

    for i, (img_batch, descr_batch) in enumerate(tqdm(dloader, leave=False)):
        img_batch=img_batch.to(config.DEVICE)
        descr_batch=descr_batch.to(config.DEVICE)

        img_batch = make_patches(img_batch, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
        
        with torch.cuda.amp.autocast():
            out = model(img_batch, descr_batch)
            print('out.shape', out.shape)
            print('descr.shape', descr_batch.shape)
            raise RuntimeError('testend')
            loss = criterion (out, descr_batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
            
    

def run():
    #vocab and tokenizer
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())

    #dataloaders
    train_dl, test_dl = get_train_test_dl(img2descr_lemma, tokenizer)
    
    #model & optim
    scaler = torch.cuda.amp.GradScaler()
    ict_model = ICTrans(n_patches= calc_num_patches(),
                        embedding_size = config.EMBED_SIZE,
                        num_heads = config.N_HEADS,
                        num_layers = config.N_TRANS_LAYERS,
                        vocab_size = vocab_size)
    ict_model = ict_model.to(config.DEVICE)
    
    optimizer = config.OPTIMIZER(ict_model.parameters(),
                                 lr = config.LR)
    loss = nn.CrossEntropyLoss()

    #logs
    '''tb_log_dir = os.path.join(config.MAIN_TB_DIR, config.get_time())
    writer = SummaryWriter(tb_log_dir)'''
    start_time= epo_start_time = datetime.now()

    #train
    for epo in range(config.EPOCHS):
        print(f'epo {epo+1}/{config.EPOCHS}')

        epo_start_time = datetime.now()

        train(model=ict_model,
              optimizer=optimizer,
              criterion  = loss,
              scaler=scaler,
              dloader=train_dl,
              #writer=writer
            )
        #lr_scheduler.step()
        
        print_elapsed_time(elapsed_time=datetime.now() - epo_start_time)

        if epo % 5 == 0:
            utils.savemodel()
    
    print_elapsed_time(elapsed_time=datetime.now() - start_time, text='model training time')


if __name__ == '__main__':
    run()