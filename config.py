import os
import torch

#


IMG_SIZE = 224
PATCH_SIZE = 64
PATCH_STRIDE = int(PATCH_SIZE*0.8)


# FOLDERS
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

