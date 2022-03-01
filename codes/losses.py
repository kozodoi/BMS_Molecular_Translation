####### LOSSES

import torch.nn as nn

def get_losses(CFG, 
               tokenizer, 
               device, 
               epoch = None):
    
    '''
    Get loss function
    '''
    
    # tests
    assert isinstance(CFG, dict), 'CFG has to be a dict with parameters'

    # training loss
    if CFG['loss_fn'] == 'CE':
        train_criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.stoi['<pad>']).to(device)
    
    # validation loss
    valid_criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.stoi['<pad>']).to(device)

    return train_criterion, valid_criterion