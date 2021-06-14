import numpy as np

import timm
from timm.utils import *

from utilities import *

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm


####### VALIDATION

def valid_epoch(loader, 
                encoder, 
                decoder, 
                tokenizer, 
                CFG, 
                device):
    
    # switch regime
    encoder.eval()
    decoder.eval()
    
    # placeholder
    TEXT_PREDS = []
    
    # inference loop
    for batch_idx, inputs in (tqdm(enumerate(loader), total = len(loader))):
        
        # extract inputs
        inputs = inputs.to(device)
        
        # forward pass
        with torch.no_grad():
            features = encoder(inputs)
            preds    = decoder.predict(features, CFG['max_len'], tokenizer)
        
        # transform preds
        seq_preds  = torch.argmax(preds, -1).detach().cpu().numpy()
        text_preds = tokenizer.predict_captions(seq_preds)
        TEXT_PREDS.append(text_preds)
        
    return np.concatenate(TEXT_PREDS)



####### VALIDATION WITH BEAM SEARCH

def valid_epoch_with_beam_search(loader, 
                                 encoder, 
                                 decoder, 
                                 tokenizer, 
                                 CFG, 
                                 device):
    
    # switch regime
    encoder.eval()
    decoder.eval()
    
    # placeholder
    TEXT_PREDS = []
    
    # decoder
    topk_decoder = TopKDecoder(decoder, CFG['beam_k'], CFG['decoder_dim'], CFG['max_len'], tokenizer)
    
    # inference loop
    for batch_idx, inputs in (tqdm(enumerate(loader), total = len(loader))):
        
        # extract inputs
        inputs = inputs.to(device)
        
        # placeholder
        seq_preds = []
        
        # forward pass
        with torch.no_grad():
            features    = encoder(inputs)
            batch_size  = features.size(0)
            encoder_dim = features.size(-1)
            features    = features.view(batch_size, -1, encoder_dim)
            h, c        = decoder.init_hidden_state(features)
            hidden      = (h.unsqueeze(0), c.unsqueeze(0))
            
            # transform preds
            decoder_outputs, decoder_hidden, other = topk_decoder(None, hidden, features)
            for b in range(batch_size):
                length     = other['topk_length'][b][0]
                tgt_id_seq = [other['topk_sequence'][di][b, 0, 0].item() for di in range(length)]
                seq_preds.append(tgt_id_seq)
              
        # transform preds
        text_preds = tokenizer.predict_captions(seq_preds)
        text_preds = [p.replace('<sos>', '') for p in text_preds]
        TEXT_PREDS.append(text_preds)
            
    return np.concatenate(TEXT_PREDS)