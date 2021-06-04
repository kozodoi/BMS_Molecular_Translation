####### AUGMENTATIONS

def get_augs(CFG, image_size = None, p_aug = None):

    # update epoch-based parameters
    if image_size is None:
        image_size = CFG['image_size']
    if p_aug is None:
        p_aug = CFG['p_aug']
        
    # normalization fior RGB images
    if CFG['num_channels'] == 3:
        if CFG['normalize']:
            if CFG['normalize'] == 'imagenet':
                CFG['pixel_mean'] = (0.485, 0.456, 0.406)
                CFG['pixels_std'] = (0.229, 0.224, 0.225)
            elif CFG['normalize'] == 'custom':
                CFG['pixel_mean'] = (0.970, 0.970, 0.970)
                CFG['pixels_std'] = (0.156, 0.156, 0.156)
        else:
            CFG['pixel_mean'] = (0, 0, 0)
            CFG['pixels_std'] = (1, 1, 1)
        
    # normalization for grayscale images
    if CFG['num_channels'] == 1:
        if CFG['normalize']:
            if CFG['normalize'] == 'imagenet':
                CFG['pixel_mean'] = (0.485)
                CFG['pixels_std'] = (0.229)
            elif CFG['normalize'] == 'custom':
                CFG['pixel_mean'] = (0.970)
                CFG['pixels_std'] = (0.156)
        else:
            CFG['pixel_mean'] = (0)
            CFG['pixels_std'] = (1)
    
    # train augmentations
    train_augs = A.Compose([A.Resize(height = image_size, 
                                     width  = image_size),
                            A.ShiftScaleRotate(p            = p_aug,
                                               shift_limit  = CFG['ssr'][0],
                                               scale_limit  = CFG['ssr'][1],
                                               rotate_limit = CFG['ssr'][2]),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixels_std']),
                            ToTensorV2()
                           ])

    # valid augmentations
    valid_augs = A.Compose([A.Resize(height  = image_size, 
                                     width   = image_size),
                            A.Normalize(mean = CFG['pixel_mean'],
                                        std  = CFG['pixels_std']),
                            ToTensorV2()
                           ])
    
    # output
    return train_augs, valid_augs