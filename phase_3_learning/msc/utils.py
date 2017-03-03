import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import time
import datetime
import codecs


class Config:
    src_vocab_size    = -1   # unk, pad, start, end
    target_vocab_size = -1
    max_source_len    = 40
    max_target_len    = 40

    batch_size        = 128
    hidden_size       = 512
    embedding_size    = 256
    num_layers        = 1
    attention         = 'bilinear'       # accepted values: [off, dot, bilinear]
    encoder_type      = 'handmade_bidirectional'    # [default, bidirectional, handmade, handmade_bidirectional]
    decoder_type      = 'argmax'    # [default]

    epochs            = 35
    dropout_rate      = 0.2
    optimizer         ='Adam' #     # [SGD, Adam, Adagrad]
    learning_rate     = 0.0003    # [1.0 for sgd, 0.0003 for adam] work well
    max_grad_norm     = 5.0
    checkpoint_dir    = 'checkpoints'
    fig_dir           = 'figs'
    result_dir        = 'results'

    x_corpus = 'corpus.en.cleaned.tok'
    x_vocab = 'vocab.15k.en'
    y_corpus = 'corpus.ja.cleaned.tok'
    y_vocab = 'vocab.15k.ja'

    train = 'train.100k'
    test = 'test.100k'
    val = 'val.100k'


def file_length(f):
    return int(os.popen('wc -l %s' % f).read().strip().split()[0])


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



class Logger():
    def __init__(self, location):
        self.f = codecs.open(location, 'w', encoding='utf-8')

    def log(self, s, show_time=True):
        if show_time:
            t = time.time()
            ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            self.f.write(ts + ': ' +  s + '\n')
        else:
            self.f.write(s + '\n')


    def close(self):
        self.f.close()





class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)






