import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Constants:
    START = "<s>"
    END = "</s>"
    UNK = "<unk>"
    PAD = "<pad>"




class Config:
    src_vocab_size = 17193 + 4   # unk, pad, start, end
    target_vocab_size = 7711 + 4 
    max_source_len = 40
    max_target_len = 40

    batch_size = 5
    hidden_size = 512
    embedding_size = 256
    num_layers = 1
    attention = 'bilinear'       # accepted values: [off, dot, bilinear]
    encoder_type = 'handmade_bidirectional'    # [default, bidirectional, handmade, handmade_bidirectional]
    decoder_type = 'argmax'    # [default]

    epochs = 35
    dropout_rate = 0.2
    optimizer='Adam' #     # [SGD, Adam, Adagrad]
    learning_rate = 0.0003    # [1.0 for sgd, 0.0003 for adam] work well
    max_grad_norm = 5.0
    checkpoint_dir = 'checkpoints_mine'

    x_corpus = 'train.en'
    x_vocab = 'vocab.en'
    y_corpus = 'train.vi'
    y_vocab = 'vocab.vi'


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


