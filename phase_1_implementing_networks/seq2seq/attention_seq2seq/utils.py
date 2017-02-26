import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class Config:
    src_vocab_size = 50000 + 1 # +1 for unk
    max_source_len = 30
    embedding_size = 256
    batch_size = 128
    hidden_size = 1024
    dropout_rate = 0.2
    num_layers = 3
    target_vocab_size = 50000 + 1 # +1 for unk
    max_target_len = 30
    learning_rate = 1.0    # sgd
    attention = 'dot'       # accepted values: [off, dot, bilinear]
    max_grad_norm = 5.0
#    checkpoint_dir = 'checkpoints_mine'
    checkpoint_dir = 'checkpoints_dillons'
#    learning_rate = 0.001  # adam


def lineplot(filename, title, xlab, ylab, curves):
    """ makes a line plot
        - title: plot title
        - filename: filename to use
        - xlab, ylab: axis labels
        - curves:  [([x1, x2, ...], name), ...]   = list of (data, name) tuples
    """
    lines, names = zip(*curves)
    for line in lines:
        plt.plot(range(len(line)), line)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(list(names))
    plt.title(title)
    plt.savefig(filename)
    plt.close()


