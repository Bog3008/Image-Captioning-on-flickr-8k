import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#import torch.nn.functional as F
#from dataset import *
import math
import os
from datetime import datetime
from tqdm import tqdm

import config
import utils
from utils import make_patches, calc_num_patches
from dataset import FlickerDM, FlickerDS, MyTokenizer, get_img2discr
from model import ICTrans, RTnet, CTnet

def print_elapsed_time(elapsed_time, text='time pre epo'):
    minutes = int(elapsed_time.total_seconds() // 60)
    seconds = int(elapsed_time.total_seconds() % 60)
    print(f'{text} {minutes}m {seconds}s')

def run(model_type='ICT'):
    model_name = utils.make_model_name()
    #vocab and tokenizer
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())

    #model & optim
    scaler = torch.cuda.amp.GradScaler()

    model_params = {"n_patches": calc_num_patches(),
               "embedding_size": config.EMBED_SIZE,
               "num_heads": config.N_HEADS,
               "num_layers": config.N_TRANS_LAYERS,
               "vocab_size": vocab_size,
               "bos_idx": tokenizer.bos_idx,
               "eos_idx": tokenizer.eos_idx,
               "dropout": config.DROPOUT,
               "tokenizer":tokenizer}
    if model_type == 'ICT':
        ict_model = ICTrans(**model_params)
    if model_type == 'RT':
        ict_model = RTnet(**model_params)
    if model_type == 'CT':
        ict_model = CTnet(**model_params)

    ict_model = ict_model.to(config.DEVICE)
    
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.17)
    #warmaper = utils.warmup_lr_sheduler(total_epochs=config.BATCH_SIZE, warmup_steps=config.WARMUP_STEPS)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmaper)

    
    if config.LOAD_MODEL:
        ict_model.load_from_checkpoint(config.LOAD_MODEL_NAME)

    callbacks = []

    if config.SAVE_MODEL:
        checkpoint_callback = ModelCheckpoint(
        monitor='train_bleu',
        dirpath='saved_models/',
        filename='model-{epoch:02d}-{train_bleu:.2f}',
        save_top_k=1,
        mode='min',
        every_n_train_steps = 10
        )
        callbacks.append(checkpoint_callback)

    logger = None
    if config.WRITE_LOGS:
        logger = WandbLogger(log_model="all", project=config.PROJECT_NAME, name=model_name)

    #logger = TensorBoardLogger("tb_logs", name="my_model")

    dm = FlickerDM(img2descr_lemma, tokenizer)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        min_epochs=1, 
        max_epochs=100, 
        precision=16,
        callbacks=callbacks,
        logger=logger#,
        #overfit_batches=1
    )
    trainer.fit(ict_model, dm)
    return 

if __name__ == '__main__':
    print('@'*50)
    print('@'*50)
    run(model_type='ICT')


  
