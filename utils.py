from typing import Any
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset import *
import math
import config
from datetime import datetime
from torchtext.data.metrics import bleu_score
from torcheval.metrics.functional.text import bleu

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
def test_onehot():
    inp = torch.randint(5, 10, (2, 22))
    out = nn.functional.one_hot(inp, 6782)
    print(out.shape)

def save_model(model, optimizer, filename):
    print('saving model...')
    filename = os.path.join(config.SAVED_MODELS_DIR, filename)
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, filename)

def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.LR

        
def make_model_name():
    current_date_and_time = datetime.now()
    formatted_string = current_date_and_time.strftime('%m_%d_%H')
    add_str = f'bs{config.BATCH_SIZE}_lr{config.LR}'
    return formatted_string+add_str

@torch.no_grad()
def img_and_descr(model, dataloader, tokenizer, batch=None, n_imgs = 2, title=None):
    if batch is None:
        batch = next(iter(dataloader))
    dataset = zip(*batch)
    fig, axs = plt.subplots(1, n_imgs, figsize=(15, 5))
    std = mean = 0.5
    model.eval()
    model.to(config.DEVICE)
    for i, (img, true_descr) in enumerate(dataset):
        if i >= n_imgs:
            break
        # feed img into model here
        img = img.to(config.DEVICE).unsqueeze(0)
        #  inference must be above<>
        tokens = model.inference(img)
        model_descr = 'model:' + ' '.join(tokenizer.decode(tokens[0]))
        #
        img = img[0]*std + mean
        img = img.cpu()
        true_descr = 'GT:' + ' '.join(tokenizer.decode(true_descr))

        final_descr = model_descr + '\n' + true_descr
        axs[i].imshow(img.permute(1, 2, 0))
        axs[i].text(1, 1, final_descr, fontsize=8, bbox=dict(facecolor='white', alpha=0.5), ha="left", va="bottom")
        axs[i].axis('off')

    if title is not None:
        plt.title(title)
    plt.show()

def calc_bleu(tokens, descr_batch, tokenizer, bleu_type=None):
        candidates = tokenizer.decode_batch(tokens)
        reference = tokenizer.decode_batch(descr_batch)
        reference = [[single_ref] for single_ref in reference]
        if bleu_type is None:
            return bleu_score(candidate_corpus=candidates, references_corpus=reference)
        else:
            if bleu_type not in [1, 2, 3, 4]:
                raise RuntimeError('Incorrect bleu number. It must be >=1 and <=4')
            weights = [0]*bleu_type
            weights[bleu_type-1] = 1
            bs = bleu_score(candidate_corpus=candidates, references_corpus=reference, max_n=bleu_type, weights=weights)
            return bs
    
class warmup_lr_sheduler:
    def __init__(self, total_epochs, warmup_steps) -> None:
        self.total_epochs = total_epochs
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos((step - self.warmup_steps) / float(max(1, (self.total_epochs+1) - self.warmup_steps)) * math.pi)))

@torch.no_grad()
def show_descr(model, dl, tokenizer, title=None):
    model.eval()
    model.to(config.DEVICE)

    batch = next(iter(dl))
    dataset = zip(*batch)
    
    for i, (img, true_descr) in enumerate(dataset):
        img = img.to(config.DEVICE).unsqueeze(0)
        #  inference must be above<>
        tokens = model.inference(img)
        model_descr = 'model:' + ' '.join(tokenizer.decode(tokens[0]))
        true_descr = 'GT:' + ' '.join(tokenizer.decode(true_descr))
        print()
        if title is not None:
            print(title)
        print('\t', model_descr)
        print('\t', true_descr)
        break


if __name__ == '__main__':
    print('-'*40)
    #test_patch_devision()
    #test_make_patches()
    test_onehot()