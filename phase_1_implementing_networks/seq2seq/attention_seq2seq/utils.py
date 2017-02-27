import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class Config:
    src_vocab_size = 17193 + 4   # unk, pad, start, end
    target_vocab_size = 7711 + 4 
    max_source_len = 40
    max_target_len = 40

    batch_size = 128
    hidden_size = 512
    embedding_size = 256
    num_layers = 1
    attention = 'bilinear'       # accepted values: [off, dot, bilinear]
    encoder_type = 'bidirectional'    # [default, bidirectional]

    epochs = 35
    dropout_rate = 0.2
    learning_rate = 1.0    # sgd
    max_grad_norm = 5.0
    checkpoint_dir = 'checkpoints_mine'


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


