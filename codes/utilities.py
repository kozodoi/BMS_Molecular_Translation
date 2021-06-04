####### UTILITIES

# competition metric
def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

# random sequences
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

# device-aware printing
def smart_print(expression, CFG):
    if CFG['device'] != 'TPU':
        print(expression)
    else:
        xm.master_print(expression)

# device-aware model save
def smart_save(weights, path, CFG):
    if CFG['device'] != 'TPU':
        torch.save(weights, path)    
    else:
        xm.save(weights, path) 

# randomness
def seed_everything(seed, CFG):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    smart_print('- setting random seed to {}...'.format(seed), CFG)
    
# torch random fix
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)