import os
import torch
import torch.optim as optim

import random
import numpy as np
#

PROJECT_NAME = 'ImageCaptioning Fliker8k'

LOAD_MODEL = False
LOAD_MODEL_NAME = r'01_25_15bs32_lr3e-05_083_bleu'
#r'01_25_17bs32_lr3e-06_091'
# for convinience it is here but I make a full path for this vsriable below
#it is assumed that  file is in the SAVED_MODELS_DIR
WRITE_LOGS = True
SAVE_MODEL =  False#True
BATCH_SIZE = 32#16 the bigger batch_size - the bigger must be embed_size
EPOCHS = 300#40#300
WARMUP_STEPS = 10

DROPOUT = 0#0.1
CLIP_VALUE = 0.1

IMG_SIZE = 224
PATCH_SIZE = 64#16#64#32#128#64
# 64 ~ 63; 32~68(on 2nd launch); 16~64; 126 ~70(not stable)
PATCH_STRIDE = int(PATCH_SIZE*0.8)


EMBED_SIZE = 512
N_HEADS = 16
N_TRANS_LAYERS = 2
MAX_SEQ_LEN = 20

NUM_WORKERS = 6
DEVICE = 'cuda'#if torch.cuda.is_available() else 'cpu'

seed=42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OPTIMIZER = optim.AdamW
LR = 3e-5#8e-5#5e-5 for ICT


# FOLDERS
IMG_FOLDER_PATH = 'data\Flickr8k_Dataset\Flicker8k_Dataset'
current_directory = os.getcwd()
train_path = r'data\Flickr8k_text\Flickr_8k.trainImages.txt'
val_path = r'data\Flickr8k_text\Flickr_8k.devImages.txt'
test_path = r'data\Flickr8k_text\Flickr_8k.testImages.txt'

train_path = os.path.join(current_directory, train_path)
val_path = os.path.join(current_directory, val_path)
test_path = os.path.join(current_directory, test_path)

def get_img_names(path:str)->list:
    with open(path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

TRAIN_IMG_NAMES = get_img_names(train_path)
VAL_IMG_NAMES = get_img_names(val_path)
TEST_IMG_NAMES = get_img_names(test_path)

DESCR_PATH =  os.path.join(current_directory, r'data\Flickr8k_text\Flickr8k.token.txt')
DESCR_LEMMA_PATH =  os.path.join(current_directory, r'data\Flickr8k_text\Flickr8k.lemma.token.txt')

img_folder_path='data\Flickr8k_Dataset\Flicker8k_Dataset'

MAIN_TB_DIR = os.path.join(current_directory, 'tb_logs')
SAVED_MODELS_DIR = os.path.join(current_directory,'saved_models')
LOAD_MODEL_NAME = os.path.join(SAVED_MODELS_DIR, LOAD_MODEL_NAME)