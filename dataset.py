from typing import Any
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import config

def show_single_img(img):
    plt.figure(figsize=(12, 6))

    mean = std = 0.5
    img = (img  * std) + mean

    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def get_img2discr(path:str, only_lowwer=True)->dict:
    '''
    path - path to txt file with line structure like this: "123.jpg#0    A sunshine."
    only_lower - use lower case transormation for words or not

    return: dict with structure: {'img_name': ['img_descr 1', 'img_descr 2'...]}
    '''
    img2descr = {}
    with open(path, 'r') as file:
        for line in file.readlines():
            parts = line.split('\t')
            img_name, img_description_id = parts[0].split('#')
            description = parts[1].rstrip().rstrip('.') #to cut off \n and dot at the end of the sentence
            if only_lowwer:
                description=description.lower()
            if img_name not in img2descr.keys():
                img2descr[img_name] = list()
            img2descr[img_name].append(description)
    return img2descr

class MyTokenizer:
    def __init__(self, img2descr, max_seq_len = config.MAX_SEQ_LEN, bos = '<bos>', eos = '<eos>', unk = '<unk>', pad = '<pad>'):
        unique_words = []
        for sentences in img2descr.values():
            for sentence in sentences: 
                unique_words.extend(sentence.split())
        self.unique_words = list(set(unique_words))
        self.unique_words += [bos, eos, unk, pad]
        self.word2token = {}
        self.token2word = {}
        for i, word in enumerate(self.unique_words):
            self.word2token[word] = i
            self.token2word[i] = word

        self.max_seq_len = max_seq_len

        self.unk_idx = self.word2token[unk]
        self.bos_idx = self.word2token[bos]
        self.eos_idx = self.word2token[eos]
        self.pad_idx = self.word2token[pad]

    def get_unique_words(self):
        return self.unique_words
    def encode(self, sentence, pad_to_max=True):
        '''encode single sentence'''
        tokenized = []
        for i, word in enumerate(sentence.split()):
            if i >= self.max_seq_len:
                break
            if word in self.word2token.keys():
                tokenized.append(self.word2token[word])
            else:
                tokenized.append(self.unk_idx)
        tokenized += [self.eos_idx]
        if pad_to_max:
            for i in range(self.max_seq_len - len(tokenized)+1): # +1 because we addad [self.eos_idx]
                tokenized.append(self.pad_idx)
        return torch.tensor([self.bos_idx] + tokenized )
    
    def decode(self, tokens):
        words = []
        for token in tokens:
            words.append(self.token2word[int(token)])
        return words
    
    def beauty_decode(self, tokens):
        tokens = [token for token in tokens if token not in [self.unk_idx, self.bos_idx, self.eos_idx, self.pad_idx]]
        return ' '.join(self.decode(tokens))

    def encode_sentences_with_padding(self, sentences):
        '''pad to all the examples for one image to make a tensor out of them.
        this function has not been tested!!!
        '''
        max_len = -1
        for sentence in sentences:
            local_len = len(sentence)
            if local_len > max_len:
                max_len = local_len
        tokenized_sentences = []
        for sentence in sentences:
            tokenized = []
            for i, word in enumerate(sentence.split()):
                if word in self.word2token.keys():
                    tokenized.append(self.word2token[word])
                else:
                    tokenized.append(self.unk_idx)
            for _ in range(max_len-len(sentence) ):
                tokenized.append(self.pad_idx)
            tokenized_sentences.apend([self.bos_idx] + tokenized + [self.eos_idx])
        return torch.tensor(tokenized_sentences)
    def probs2words(self, probs_t):
        '''
        probs_t - 2dim  tensor (seq_len, vocab_size) where each "row" probs for words

        return - list of strs
        '''
        description = []
        _, tokens = torch.max(probs_t, dim=1)
        #print(tokens.shape)
        
        for token in tokens:
            description.append(self.token2word[int(token)])
        return description

def tokenize_img2descr(img2descr, tokenizer):
    img2tok_list = {}
    for img_name, descr_list in img2descr.items():
        #descr_list[0] - i use only one description for each image. use different disription cause too many optimization issues
        img2tok_list[img_name] = tokenizer.encode(descr_list[0])

    return img2tok_list

class FlickerDS(Dataset):
    def __init__(self, img_folder_path, img2descr, img_names, img_size, tokenizer, transform=None):
        
        self.img_folder_path = img_folder_path
        self.img_names = img_names

        self.img2tok_list = tokenize_img2descr(img2descr, tokenizer)
        self.tokenizer = tokenizer

        self.img_size = img_size

        if transform is None:
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],), 
                            transforms.Resize((self.img_size, self.img_size), antialias=True)
                        ])
        self.transform = transform

    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, indx):
        decription_list = self.img2tok_list[self.img_names[indx]]
        img_full_path =  os.path.join(self.img_folder_path, self.img_names[indx])
        img = self.transform(Image.open(img_full_path).convert("RGB"))
        return img, decription_list

def test_dataset():
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    
    ds = FlickerDS(img_folder_path='data\Flickr8k_Dataset\Flicker8k_Dataset',
                   img2descr=img2descr_lemma,
                   img_names=config.TEST_IMG_NAMES,
                   img_size = 256,
                   tokenizer=tokenizer)
    img, descr = ds[random.randint(0, 5)]
    #show_single_img(img)
    print(descr)
    print(tokenizer.decode(descr)) #for 1 descr per image
    print(tokenizer.beauty_decode(descr))
    #print(*list(map(tokenizer.decode, descr)),sep='\n' ) - for 5 descr per image

def test_tokenizer():
    img2descr_lemma = get_img2discr(config.DESCR_LEMMA_PATH)
    tokenizer = MyTokenizer(img2descr_lemma)
    for word in ['sun shine', '1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 11 12 13']:
        t_w = tokenizer.encode(word)
        print(f'source len:{len(word.split())}, enc len:{len(t_w)}')
        print(f'source :{word.split()}, enc&dec :{tokenizer.decode(t_w)}')

if __name__ == '__main__':
    print('-'*40)
    test_dataset()
    #test_tokenizer()
        