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

def get_train_test_dl(img2descr_lemma, tokenizer, shuffle_train=True, shuffle_test=True):
    train_ds = FlickerDS(img_folder_path=config.IMG_FOLDER_PATH,
                   img2descr=img2descr_lemma,
                   img_names=config.TRAIN_IMG_NAMES,
                   img_size = config.IMG_SIZE,
                   tokenizer=tokenizer)
    train_dl = DataLoader(train_ds,
                    batch_size= config.BATCH_SIZE, 
                    shuffle=shuffle_train, 
                    #num_workers=config.NUM_WORKERS,
                    pin_memory=True)
    
    test_ds = FlickerDS(img_folder_path=config.IMG_FOLDER_PATH,
                   img2descr=img2descr_lemma,
                   img_names=config.TEST_IMG_NAMES,
                   img_size = config.IMG_SIZE,
                   tokenizer=tokenizer)
    test_dl = DataLoader(test_ds,
                    batch_size= config.BATCH_SIZE, 
                    shuffle=shuffle_test, 
                    #num_workers=config.NUM_WORKERS,
                    pin_memory=True)
    return train_dl, test_dl
def print_elapsed_time(elapsed_time, text='time pre epo'):
    minutes = int(elapsed_time.total_seconds() // 60)
    seconds = int(elapsed_time.total_seconds() % 60)
    print(f'{text} {minutes}m {seconds}s')

def train(model, optimizer, criterion , scaler, dloader, tokenizer):
    model.train()
    avg_loss = 0
    losses = []
    for img_batch, descr_batch in tqdm(dloader, leave=False):
        img_batch=img_batch.to(config.DEVICE)
        descr_batch=descr_batch.to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            out = model(img_batch, descr_batch[:, torch.randint(low=0, high=5, size=(1,))].squeeze(1))
            bs, seq_len, n_clas = out.shape
            padd_tensor = torch.full((bs, 1), tokenizer.pad_idx).to(config.DEVICE)
            loss = 0
            for single_descr in descr_batch.permute(1, 0, 2):
                single_descr = torch.cat((single_descr[:,1:], padd_tensor), dim=1)
                loss += criterion(out.view(seq_len*bs, n_clas), 
                                single_descr.view(seq_len*bs))
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)
            losses.append(loss)
            avg_loss += loss
    '''if math.isnan(avg_loss):
        print(losses)
        raise RuntimeError('test: Nan loss')'''

    return avg_loss/len(dloader)       

@torch.no_grad()
def evaluate(model, criterion, dloader, tokenizer):
    model.eval()
    avg_loss = 0
    avg_bleu = 0
    

    for img_batch, descr_batch in tqdm(dloader, leave=False):
        img_batch=img_batch.to(config.DEVICE)
        descr_batch=descr_batch.to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            #print(descr_batch.shape)#<>
            #print(descr_batch[:, torch.randint(low=0, high=5, size=(1,))].shape)
            out = model(img_batch, descr_batch[:, torch.randint(low=0, high=5, size=(1,))].squeeze(1))
            bs, seq_len, n_clas = out.shape
            padd_tensor = torch.full((bs, 1), tokenizer.pad_idx).to(config.DEVICE)
            loss = 0
            for single_descr in descr_batch.permute(1, 0, 2):
                #print('sd', single_descr.shape)
                single_descr = torch.cat((single_descr[:,1:], padd_tensor), dim=1)
                loss += criterion(out.view(seq_len*bs, n_clas), 
                                single_descr.view(seq_len*bs))
            avg_loss += loss
            #calc forward bleu
            tokens = torch.argmax(out, dim=2)
            avg_bleu += utils.calc_bleu(tokens, descr_batch, tokenizer)
            #calc inference bleu
            '''if use_inference:
                tokens = model.inference(img_batch)
                avg_infer_bleu += utils.calc_bleu(tokens, descr_batch, tokenizer)'''
    return avg_loss/len(dloader), avg_bleu/len(dloader)

@torch.no_grad()
def evaluate_iference(model, dloader, tokenizer, n_examples = 1001):
    model.eval()
    avg_bleu = 0
    i_batches_passed=1
    #padd_tensor = torch.full((config.BATCH_SIZE, 1), tokenizer.pad_idx).to(config.DEVICE)

    for i ,(img_batch, descr_batch) in enumerate(tqdm(dloader, leave=False)):
        if i > (n_examples /config.BATCH_SIZE):
            # to not break dloader. if we do break -> dloader on the next iter wont start from begining
            continue
        i_batches_passed = i
        img_batch=img_batch.to(config.DEVICE)
        descr_batch=descr_batch.to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            
            tokens = model.inference(img_batch)
            avg_bleu += utils.calc_bleu(tokens, descr_batch, tokenizer)
            #descr_batch = torch.cat((descr_batch[:,1:], padd_tensor), dim=1)
            
    return avg_bleu/i_batches_passed



def train_one_batch(model, optimizer, criterion , scaler, batch, tokenizer):
    model.train()
    avg_loss = 0
    avg_bleu = 0
    img_batch, descr_batch = batch
    img_batch=img_batch.to(config.DEVICE)
    descr_batch=descr_batch.to(config.DEVICE)

    #img_batch = make_patches(img_batch, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
    #padd_tensor = torch.full((config.BATCH_SIZE, 1), tokenizer.pad_idx).to(config.DEVICE)
    with torch.cuda.amp.autocast():
        out = model(img_batch, descr_batch)
        bs, seq_len, n_clas = out.shape
        padd_tensor = torch.full((bs, 1), tokenizer.pad_idx).to(config.DEVICE)   
        descr_batch = torch.cat((descr_batch[:,1:], padd_tensor), dim=1)
        loss = criterion(out.view(seq_len*bs, n_clas), 
                            descr_batch.view(seq_len*bs))
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
        tokens = model.inference(img_batch) #<>
        #tokens = torch.argmax(out, dim=2)
        avg_bleu += utils.calc_bleu(tokens, descr_batch, tokenizer)
        #candidates = tokenizer.decode_batch(tokens)
        #reference = tokenizer.decode_batch(descr_batch)
        #reference_p = [[single_ref] for single_ref in reference]
        #print('candidates', ' '.join(candidates[0]))
        #print('reference', ' '.join(reference[0]))
        '''for single_cand, singl_ref in zip(candidates, reference):
            print('candidates', ' '.join(single_cand))
            print('reference', ' '.join(singl_ref))'''
        
        #avg_bleu += bleu_score(candidate_corpus=candidates, references_corpus=reference)
        avg_loss += loss
        
    return avg_loss, avg_bleu

def run_train_one_batch(local_epochs, model_type='ICT'):
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())

    #dataloaders
    train_dl, test_dl = get_train_test_dl(img2descr_lemma, tokenizer, shuffle_test=True)
    
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
    
    

    batch = next(iter(test_dl))

    losses = []
    bleu_scores = []
    #train
    for epo in tqdm(range(local_epochs)):

        epo_loss, epo_bleu = train_one_batch(model=ict_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        scaler=scaler,
                        batch=batch,
                        tokenizer=tokenizer
                        #writer=writer
                        )
        #lr_scheduler.step()
        losses.append(float(epo_loss))
        bleu_scores.append(float(epo_bleu))
    utils.img_and_descr(ict_model, test_dl, tokenizer, batch=batch)
    print('Losses:', losses)
    print('Bleu', bleu_scores)
def run(model_type='ICT'):
    model_name = utils.make_model_name()
    #vocab and tokenizer
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    vocab_size = len(tokenizer.get_unique_words())

    #dataloaders
    train_dl, test_dl = get_train_test_dl(img2descr_lemma, tokenizer)
    
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
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.17)
    #warmaper = utils.warmup_lr_sheduler(total_epochs=config.BATCH_SIZE, warmup_steps=config.WARMUP_STEPS)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmaper)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, size_average=True)
    
    if config.LOAD_MODEL:
        #filename = os.path.join(config.SAVED_MODELS_DIR, filename)
        print('loading model...')
        utils.load_model(ict_model, optimizer, config.LOAD_MODEL_NAME)
        utils.show_descr(model=ict_model, dl=train_dl, tokenizer=tokenizer, title='train sentence comparison')
        #train_infer_bleu = evaluate_iference( model=ict_model, dloader=train_dl, tokenizer=tokenizer)
        #print('train_infer_bleu', train_infer_bleu)
        #utils.img_and_descr(ict_model, train_dl, tokenizer, n_imgs=3)
        test_loss, test_bleu = evaluate(model=ict_model, criterion=criterion, dloader=test_dl, tokenizer=tokenizer)
        print('loss bleu', test_loss, test_bleu)
        return

    #logs
    if config.WRITE_LOGS:
        tb_log_dir = os.path.join(config.MAIN_TB_DIR, utils.get_time())
        writer = SummaryWriter(tb_log_dir)

    start_time= epo_start_time = datetime.now()

    best_test_loss = float('+inf')
    test_loss, test_bleu, test_inference_bleu = None, None, None
    train_infer_bleu = None
    best_test_inference_bleu = 0
    epo_loss = 0
    save_cool_down = 0
    #train
    for epo in range(config.EPOCHS):
        os.system('cls')
        print(f'epo {epo+1}/{config.EPOCHS}')
        print(f'train avg loss: {epo_loss}')
        print(f'train inferene bleu: {train_infer_bleu}')
        print(f'epo: {epo}; test_loss: {test_loss}; test_bleu: {test_bleu}')
        print(f'test inferene bleu: {test_inference_bleu}')
        print_elapsed_time(elapsed_time=datetime.now() - epo_start_time)
        print('LR:', optimizer.param_groups[0]['lr'])

        utils.show_descr(model=ict_model, dl=train_dl, tokenizer=tokenizer, title='train sentence comparison')
        utils.show_descr(model=ict_model, dl=test_dl, tokenizer=tokenizer, title='test sentence comparison')
        
        epo_start_time = datetime.now()
        epo_loss = train(model=ict_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        scaler=scaler,
                        dloader=train_dl, 
                        tokenizer=tokenizer
                        )
        #lr_scheduler.step()
        test_loss, test_bleu = evaluate(model=ict_model, criterion=criterion, dloader=test_dl, tokenizer=tokenizer)
        

        if epo % 2:
            test_inference_bleu = evaluate_iference(model=ict_model, dloader=test_dl, tokenizer=tokenizer)
            train_infer_bleu = evaluate_iference(model=ict_model, dloader=train_dl, tokenizer=tokenizer)
            #if config.SAVE_MODEL and (test_inference_bleu > 0) and (test_inference_bleu > best_test_inference_bleu):
            #    utils.save_model(ict_model, optimizer, model_name+'_BLEU')
            if config.WRITE_LOGS:
                writer.add_scalar('Metrics/bleu_test_inference', test_inference_bleu, epo)
                writer.add_scalar('Metrics/bleu_train_inference', train_infer_bleu, epo)
        if config.WRITE_LOGS:
            writer.add_scalar('Loss/train', epo_loss, epo)
            writer.add_scalar('Loss/test', test_loss, epo)
            writer.add_scalar('Metrics/bleu_test_forward', test_bleu, epo)

    
        if config.SAVE_MODEL and ((epo % 10 == 0) or ((epo+1) == config.BATCH_SIZE)):#test_loss < best_test_loss:
            utils.save_model(ict_model, optimizer, model_name)
        try:
            if config.SAVE_MODEL and train_infer_bleu >= 0.83:
                if save_cool_down == 0:
                    utils.save_model(ict_model, optimizer, model_name+'_083_bleu')
                    save_cool_down = 8
                save_cool_down -= 1
        except:
            pass
    
    print_elapsed_time(elapsed_time=datetime.now() - start_time, text='model training time')

    utils.img_and_descr(ict_model, train_dl, tokenizer, n_imgs=4)
    utils.img_and_descr(ict_model, test_dl, tokenizer, n_imgs=4)

if __name__ == '__main__':
    print('@'*50)
    print('@'*50)
    run(model_type='ICT')
    #run_train_one_batch(local_epochs=100, model_type='ICT') # added inference bleu


# ICT - params
#dubug info
'''
print(type(out))
print(f'out type: {out.dtype}')
print(f'out device: {out.device}')
print(f'out grad: {out.requires_grad}')
print('out.shape', out.shape)

print(f'descr type: {descr_batch.dtype}')
print(f'descr device: {descr_batch.device}')
print(f'descr grad: {descr_batch.requires_grad}')
print('descr.shape', descr_batch.shape)
raise RuntimeError('testend')'''