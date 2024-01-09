import os
import torch
import torch.optim as optim
#


IMG_SIZE = 224
PATCH_SIZE = 64
PATCH_STRIDE = int(PATCH_SIZE*0.8)

BATCH_SIZE = 2
NUM_WORKERS = 6
EMBED_SIZE = 256
N_HEADS = 4
N_TRANS_LAYERS = 6
MAX_SEQ_LEN = 20

DEVICE = 'cuda'#if torch.cuda.is_available() else 'cpu'
OPTIMIZER = optim.AdamW
LR = 1e-3
EPOCHS = 5

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
SAVED_MODELS_DIR = 'saved_models'
