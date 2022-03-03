import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
import pandas as pd

from utilities import smart_print
from augmentations import get_augs


####### DATASETS

class ImageData(Dataset):
    
    '''
    Image train/validation dataset class
    '''
    
    def __init__(self, 
                 df, 
                 tokenizer  = None, 
                 channels   = 3,
                 crop       = False, 
                 padding    = False,
                 morphology = False,
                 meta       = False,
                 transform  = None):
        super().__init__()
        self.df         = df
        self.tokenizer  = tokenizer
        self.file_paths  = df['file_path'].values
        self.labels     = df['InChI_text'].values
        self.transform  = transform
        self.crop       = crop
        self.channels   = channels
        self.morphology = morphology
        self.meta       = meta
        self.padding    = padding
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # import
        file_path = self.file_paths[idx]        
        image    = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
        if image is None:
            raise FileNotFoundError(file_path)
            
        # image meta data
        if self.meta:
            meta_area  = (image.shape[0] * image.shape[1]) / 2500000
            meta_ratio = (image.shape[0] / image.shape[1]) / 30.0
            meta       = torch.LongTensor([meta_area, meta_ratio])
            
        # morphological transforms
        if self.morphology:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2, 2)))
            image = cv2.erode(image, np.ones((2, 2)))
                        
        # smart crop
        if self.crop:
            image = smart_crop(image)
            
        # convert to RGB
        if self.channels == 3:
            image = cv2.merge([image, image, image]).astype(np.float32)
        elif self.channels == 1:
            image = image.astype(np.float32)
            
        # padding
        if self.padding:
            image = pad_image(image)
        
        # augmentations
        if self.transform:
            image = self.transform(image = image)['image']
            
        # output
        label        = torch.LongTensor(self.tokenizer.text_to_sequence(self.labels[idx]))
        label_length = torch.LongTensor([len(label)])
        if self.meta:
            return image, meta, label, label_length
        else:
            return image, label, label_length
        
        
        
class ImageTestData(Dataset):
    
        
    '''
    Image test dataset class
    '''
    
    def __init__(self, 
                 df, 
                 channels   = 3,
                 crop       = False, 
                 padding    = False,
                 morphology = False,
                 meta       = False,
                 transform  = None):
        super().__init__()
        self.df           = df
        self.file_paths    = df['file_path'].values
        self.transform    = transform
        self.crop         = crop
        self.channels     = channels
        self.padding      = padding
        self.morphology   = morphology
        self.meta         = meta
        self.fix_transform = A.Compose([A.Transpose(p = 1), A.VerticalFlip(p = 1)])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # import
        file_path = self.file_paths[idx]
        image    = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) 
        if image is None:
            raise FileNotFoundError(path)
            
        # image meta data
        if self.meta:
            meta_area = (image.shape[0] * image.shape[1]) / 2500000
            if image.shape[0] > image.shape[1]:
                meta_ratio = (image.shape[0] / image.shape[1]) / 30.0
            else:
                meta_ratio = (image.shape[1] / image.shape[0]) / 30.0
            meta = torch.LongTensor([meta_area, meta_ratio])
            
        # morphological transforms
        if self.morphology:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((2, 2)))
            image = cv2.erode(image, np.ones((2, 2)))
            
        # smart crop
        if self.crop:
            image = smart_crop(image)
            
        # convert to RGB
        if self.channels == 3:
            image = cv2.merge([image, image, image]).astype(np.float32)
        elif self.channels == 1:
            image = image.astype(np.float32)
        
        # fix rotation
        h, w = image.shape[0], image.shape[1]
        if h > w:
            image = self.fix_transform(image = image)['image']
            
        # padding
        if self.padding:
            image = pad_image(image)
        
        # augmentations
        if self.transform:
            image = self.transform(image = image)['image']
            
        # output    
        if self.meta:
            return image, meta
        else:
            return image



####### BATCH COLLATE HELPER FUNCTION

''' 
Borrowed from https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-starter
'''

def bms_collate(batch, tokenizer):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first = True, padding_value = tokenizer.stoi['<pad>'])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)



####### DATA PREP

def get_data(df, 
             fold, 
             CFG, 
             epoch = None):
    
    '''
    Get training and validation data
    '''
    
    # tests
    assert isinstance(df,   pd.DataFrame), 'df has to be a pandas dataframe'
    assert isinstance(fold, int),          'fold has to be an integer'
    assert isinstance(CFG,  dict),         'CFG has to be a dict with parameters'
    
    # epoch number
    if epoch is None:
        epoch = 0

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)
    if CFG['valid_subset']:
        df_valid = df_valid.head(CFG['valid_subset'])
    smart_print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)
    
    # checks
    assert len(df_train) + len(df_valid) == len(df), 'Wrong number of observations'
        
    # extra data
    if CFG['data_ext']:
        df_extra_epoch = df_extra.sample(n = CFG['data_ext'], random_state = CFG['seed'] + epoch).reset_index(drop = True)
        df_train       = pd.concat([df_train, df_extra_epoch], axis = 0).reset_index(drop = True)
        smart_print('- appending extra data to train...', CFG)
        smart_print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)

    # subset for debug mode
    if CFG['debug']:
        df_train = df_train.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        smart_print('- subsetting data for debug mode...', CFG)
        smart_print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)), CFG)
        
    # sort validation data for efficiency
    df_valid['InChI_length'] = df_valid['InChI'].str.len()
    df_valid = df_valid.sort_values(by = 'InChI_length', ascending = False).reset_index(drop = True)
    del df_valid['InChI_length']
    
    return df_train, df_valid




####### DATA LOADERS

from utilities import *

def get_loaders(df_train, 
                df_valid, 
                tokenizer, 
                CFG, 
                epoch = None):
    
    '''
    Get training and validation dataloaders
    '''
    
    ##### PREPARATIONS
    
    # tests
    assert isinstance(df_train, pd.DataFrame), 'df_train has to be a pandas dataframe'
    assert isinstance(df_valid, pd.DataFrame), 'df_valid has to be a pandas dataframe'
    assert isinstance(CFG, dict),              'CFG has to be a dict with parameters'

    # epoch-based params
    image_size = CFG['image_size']
    p_aug      = CFG['p_aug']


    ##### DATASETS
        
    # augmentations
    train_augs, valid_augs = get_augs(CFG, image_size, p_aug)

    # datasets
    train_dataset = ImageData(df         = df_train, 
                              transform  = train_augs,
                              tokenizer  = tokenizer, 
                              channels   = CFG['num_channels'],
                              crop       = CFG['smart_crop'],
                              morphology = CFG['morphology'],
                              padding    = CFG['padding'],
                              meta       = CFG['meta_data'])
    valid_dataset = ImageTestData(df         = df_valid, 
                                  transform  = valid_augs,
                                  channels   = CFG['num_channels'],
                                  crop       = CFG['smart_crop'],
                                  morphology = CFG['morphology'],
                                  padding    = CFG['padding'],
                                  meta       = CFG['meta_data'])
    
    
    ##### DATA SAMPLERS
    
    # samplers
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
        
    ##### DATA LOADERS
    
    # data loaders
    train_loader = DataLoader(dataset        = train_dataset, 
                              batch_size     = CFG['batch_size'], 
                              shuffle          = True,
                              num_workers    = CFG['cpu_workers'],
                              drop_last      = True, 
                              collate_fn     = lambda b: bms_collate(b, tokenizer),
                              worker_init_fn = worker_init_fn,
                              pin_memory     = False)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['valid_batch_size'], 
                              shuffle       = False,
                              num_workers = CFG['cpu_workers'],
                              drop_last   = False,
                              pin_memory  = False)
    
    # feedback
    smart_print('- image size: {}x{}, p(augment): {}'.format(image_size, image_size, p_aug), CFG)
    if epoch is None:
        smart_print('-' * 55, CFG)
    
    return train_loader, valid_loader