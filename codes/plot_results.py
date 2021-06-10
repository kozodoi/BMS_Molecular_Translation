import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


####### PLOTTING

def plot_results(trn_losses, val_scores, fold, CFG):

    # plot loss lines
    plt.figure(figsize = (20, 8))
    plt.plot(range(1, CFG['num_epochs'] + 1), trn_losses, color = 'red',   label = 'Train Loss')

    # annotations
    plt.ylabel('Loss', size = 14); plt.xlabel('Epoch', size = 14)
    plt.legend(loc = 2, fontsize = 'large')

    # plot metric line
    plt2 = plt.gca().twinx()
    plt2.plot(range(1, CFG['num_epochs'] + 1), val_scores, color = 'blue', label = 'Valid Score')

    # plot points with the best metric
    x = np.argmin(np.array(val_scores)) + 1; y = min(val_scores)
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x, y, s = 200, color = 'blue')
    plt.text(x - 0.03*xdist, y - 0.050*ydist, 'score = {:.2f}'.format(y), size = 15)

    # annotations
    plt.ylabel('AUC', size = 14)
    plt.title('Fold {} Performance'.format(fold + 1), size = 18)
    plt.legend(loc = 3, fontsize = 'large')

    # export
    plt.savefig(CFG['out_path'] + 'fig_perf_fold{}.png'.format(fold))
    plt.show()