####### MODEL PREP

from encoder import Encoder
from decoder import _inflate, Attention, DecoderWithAttention, TopKDecoder
import timm
import torch
import gc


def get_model(CFG, 
              device, 
              pretrained = None):
        
    ##### INSTANTIATE MODEL
    
    # pretrained weights
    if pretrained is None:
        pretrained = CFG['pretrained']

    # CNN encoder
    encoder = Encoder(backbone     = CFG['backbone'], 
                      num_channels = CFG['num_channels'],
                      pretrained   = True)
    
    # RNN decoder
    decoder = DecoderWithAttention(attention_dim = CFG['attention_dim'],
                                   embed_dim     = CFG['embed_dim'],
                                   encoder_dim   = CFG['encoder_dim'],
                                   decoder_dim   = CFG['decoder_dim'],
                                   vocab_size    = CFG['len_tokenizer'],
                                   dropout       = CFG['dropout'],
                                   device        = device)
    
    ##### PRETRAINED WEIGHTS 
    
    if pretrained:
        if pretrained != 'imagenet':
            
            # import states
            states = torch.load(pretrained, map_location = torch.device('cpu'))
            
            # load weights
            encoder.load_state_dict(states['encoder'])
            print('-- custom encoder weights loaded')
            decoder.load_state_dict(states['decoder'])
            print('-- custom decoder weights loaded')

            # clear memory
            del states
            gc.collect()
        
    return encoder, decoder