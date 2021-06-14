import timm
from timm.utils import *

from utilities import *

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm



####### TRAINING

def train_epoch(loader, 
                encoder, 
                decoder, 
                encoder_optimizer, 
                decoder_optimizer,
                encoder_scheduler, 
                decoder_scheduler,
                criterion, 
                autocast,
                scaler,
                epoch, 
                CFG, 
                device):
       
    # switch regime
    encoder.train()
    decoder.train()
    
    # running loss
    trn_loss = AverageMeter()
    
    # loader length
    len_loader = CFG['max_batches'] if CFG['max_batches'] else len(loader) 

    # update scheduler on epoch
    if not CFG['update_on_batch']:
        encoder_scheduler.step() 
        decoder_scheduler.step() 
        if epoch == CFG['warmup']:
            encoder_scheduler.step() 
            decoder_scheduler.step() 

    # loop through batches
    for batch_idx, (inputs, labels, lengths) in (tqdm(enumerate(loader), total = len_loader)):

        # extract inputs and labels
        inputs  = inputs.to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)
        
        # update scheduler on batch
        if CFG['update_on_batch']:
            encoder_scheduler.step(epoch + 1 + batch_idx / len_loader)
            decoder_scheduler.step(epoch + 1 + batch_idx / len_loader)
            
        # passes and weight updates
        with torch.set_grad_enabled(True):
            
            # forward pass 
            with autocast():
                features = encoder(inputs)
                preds, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels, lengths)
                targets = caps_sorted[:, 1:]
                preds   = pack_padded_sequence(preds,   decode_lengths, batch_first = True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first = True).data
                loss    = criterion(preds, targets)
                
            # backward pass
            if CFG['use_amp'] and CFG['device'] == 'GPU':
                scaler.scale(loss).backward()   
            else:
                loss.backward() 
                
            # gradient clipping
            if CFG['grad_clip']:
                encoder_grad_norm = clip_grad_norm_(encoder.parameters(), CFG['grad_clip'])
                decoder_grad_norm = clip_grad_norm_(decoder.parameters(), CFG['grad_clip'])

            # update weights
            if ((batch_idx + 1) % CFG['accum_iter'] == 0) or ((batch_idx + 1) == len(loader)):
                if CFG['use_amp'] and CFG['device'] == 'GPU':
                    scaler.step(encoder_optimizer)
                    scaler.step(decoder_optimizer)
                    scaler.update()
                else:
                    encoder_optimizer.step()
                    decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
                
        # update loss
        trn_loss.update(loss.item() * CFG['accum_iter'], inputs.size(0))

        # feedback
        if CFG['batch_verbose']:
            if (batch_idx > 0) and (batch_idx % CFG['batch_verbose'] == 0):
                smart_print('-- batch {} | cur_loss = {:.6f}, avg_loss = {:.6f}'.format(
                    batch_idx, loss.item(), trn_loss.avg), CFG)
                
        # early stop
        if (batch_idx == len_loader):
            break
        
        # clear memory
        '''
        del inputs, labels, preds, loss, targets, features, caps_sorted, decode_lengths, alphas, sort_ind
        gc.collect()
        '''
        
    return trn_loss.sum