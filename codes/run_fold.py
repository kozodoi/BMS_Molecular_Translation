from utilities import *
from data import get_loaders
from optimizers import get_optimizer, get_scheduler
from losses import get_losses
from training import train_epoch
from validation import valid_epoch
import time
import gc


####### WRAPPER FUNCTION

def run_fold(fold, df_trn, df_val, CFG, encoder, decoder, tokenizer, autocast, scaler, device):

    ##### PREPARATIONS
    
    # reset seed
    seed_everything(CFG['seed'] + fold - 1, CFG)
    
    # update device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # get optimizers
    encoder_optimizer = get_optimizer(CFG, encoder, CFG['cnn_eta'])
    decoder_optimizer = get_optimizer(CFG, decoder, CFG['rnn_eta'])

    # get schedulers
    encoder_scheduler = get_scheduler(CFG, encoder_optimizer)
    decoder_scheduler = get_scheduler(CFG, decoder_optimizer)
    
    # get loaders
    trn_loader, val_loader = get_loaders(df_trn, df_val, tokenizer, CFG)
    
    # get valid labels
    val_labels = df_val['InChI'].values
        
    # placeholders
    trn_losses  = []
    val_scores  = []
    lrs_encoder = []
    lrs_decoder = []
    
    
    ##### TRAINING AND INFERENCE

    for epoch in range(CFG['num_epochs']):
                
        ### PREPARATIONS

        # timer
        epoch_start = time.time()
            
        # get losses            
        trn_criterion, val_criterion = get_losses(CFG, tokenizer, device, epoch)
        
        # update train loader
        if CFG['data_ext']:
            df_trn, _     = get_data(df, fold, CFG, epoch)  
            trn_loader, _ = get_loaders(df_trn, df_val, tokenizer, CFG, epoch)


        ### MODELING
        
        # training
        gc.collect()
        trn_loss = train_epoch(loader            = trn_loader, 
                               encoder           = encoder, 
                               decoder           = decoder, 
                               encoder_optimizer = encoder_optimizer, 
                               decoder_optimizer = decoder_optimizer, 
                               encoder_scheduler = encoder_scheduler,
                               decoder_scheduler = decoder_scheduler,
                               criterion         = trn_criterion, 
                               autocast          = autocast,
                               scaler            = scaler,
                               epoch             = epoch,
                               CFG               = CFG,
                               device            = device)
        
        # inference
        gc.collect()
        val_preds = valid_epoch(loader    = val_loader, 
                                encoder   = encoder, 
                                decoder   = decoder,       
                                tokenizer = tokenizer,
                                CFG       = CFG,
                                device    = device)


        ### EVALUATION
        
        # data length
        len_df_trn = CFG['max_batches'] * CFG['batch_size'] if CFG['max_batches'] else len(df_trn) 
       
        # reduce losses
        trn_loss   = trn_loss / len_df_trn
        lr_encoder = encoder_scheduler.state_dict()['_last_lr'][0]
        lr_decoder = decoder_scheduler.state_dict()['_last_lr'][0]   
            
        # compute validation score
        val_preds = [f'InChI=1S/{text}' for text in val_preds]
        val_score = get_score(val_labels, val_preds)  

        # save LR and losses
        lrs_encoder.append(lr_encoder)
        lrs_decoder.append(lr_decoder)
        trn_losses.append(trn_loss)
        val_scores.append(val_score)
        
        # feedback
        smart_print('-- epoch {}/{} | lr = {:.6f} / {:.6f} | trn_loss = {:.4f} | val_score = {:.2f} | {:.2f} min'.format(
            epoch + 1, CFG['num_epochs'], lrs_encoder[epoch], lrs_decoder[epoch],
            trn_losses[epoch], val_scores[epoch],
            (time.time() - epoch_start) / 60), CFG)
        
        # send metrics to neptune
        if CFG['tracking']:
            neptune.send_metric('trn_loss{}'.format(fold),        trn_losses[epoch])
            neptune.send_metric('val_score{}'.format(fold),       val_scores[epoch])
            neptune.send_metric('score{}_{}'.format(fold, epoch), val_scores[epoch])
        
        # export weights and save preds
        if val_scores[epoch] <= min(val_scores):
            val_preds_best = val_preds.copy()
            smart_save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
                       CFG['out_path'] + 'weights_fold{}.pth'.format(fold), CFG)
        if CFG['save_all']:
            smart_save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 
                       CFG['out_path'] + 'weights_fold{}_epoch{}.pth'.format(fold, epoch), CFG)      
    
    return trn_losses, val_scores, val_preds_best