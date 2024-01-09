import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset import *
import math
import config
from datetime import datetime

# Define function to split image into patches
# config.PATCH_SIZE - x and y size of patch; config.PATCH_STRIDE - step in pixels  
def make_patches(img_t, size=16, stride=10):
    return img_t.unfold(2, size, stride).unfold(3, size, stride).squeeze(dim=4)#.unfold(4, size, stride).squeeze(dim=4)

def calc_num_patches():
    input = torch.randn([1, 3, config.IMG_SIZE, config.IMG_SIZE])
    img_p = make_patches(input, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
    bs, chanels, x_patch, y_patch, x, y = img_p.shape
    assert x_patch == y_patch, 'x_patch != y_patch, there is smth wrong in image size maybe x_img_size != y_img_size'
    return x_patch
# test functions for 'make_patches'
def plot_patches(img_p):
    mean = std = 0.5
    img_p = (img_p  * std) + mean
    bs, chanels, x_patch, y_patch, x, y = img_p.shape 
    fig, axses = plt.subplots(x_patch, y_patch)
    for i in range(x_patch):
        for j in range(y_patch):
            axses[i, j].imshow(img_p[:,:,i, j].squeeze(0).permute(1, 2, 0))
            axses[i, j].axis('off')  # Turn off axis labels

    plt.show()
def test_patch_devision():
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    
    ds = FlickerDS(img_folder_path='data\Flickr8k_Dataset\Flicker8k_Dataset',
                   img2descr=img2descr_lemma,
                   img_names=config.TEST_IMG_NAMES,
                   img_size = 256,
                   tokenizer=tokenizer)
    img, descr = ds[2]
    
    img = img.unsqueeze(0)
    img_p = make_patches(img, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
    print('shape before', img.shape)
    print('shape after', img_p.shape)
    
    #lets take a look
    plot_patches(img_p)

def test_make_patches():
    #I forgot about the function above
    img = torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
    ooo = make_patches(img, size=config.PATCH_SIZE, stride=config.PATCH_STRIDE)
    print(ooo.shape)
    print(calc_num_patches())
    
#
def get_time():
    current_datetime = datetime.now()
    return current_datetime.strftime('%Y_%m_%d_%Hh%Mm')
#

if __name__ == '__main__':
    print('-'*40)
    #test_patch_devision()
    test_make_patches()