"""
TODO - DOCUMENTATION

"""
import sys; sys.path.append('../')    # sigh 
from msc.constants import Constants
import numpy as np
import os


class Dataset(object):
    def __init__(self, config, data_root):
        self.batch_size = config.batch_size
        self.max_source_len = config.max_source_len
        self.max_target_len = config.max_target_len

        self.data_root = data_root
        lx_corpus = os.path.join(self.data_root, config.x_corpus)
        lx_vocab = os.path.join(self.data_root, config.x_vocab)
        ly_corpus = os.path.join(self.data_root, config.y_corpus)
        ly_vocab = os.path.join(self.data_root, config.y_vocab)

        self.indices = {
            'train': [int(x.strip()) for x in open(os.path.join(self.data_root, config.train))]
            'test': [int(x.strip()) for x in open(os.path.join(self.data_root, config.test))]
            'val': [int(x.strip()) for x in open(os.path.join(self.data_root, config.val))]
        }

        self.batch_index = 0

        self.lx_w_to_i, self.lx_i_to_w = self.parse_vocab(lx_vocab)
        self.ly_w_to_i, self.ly_i_to_w = self.parse_vocab(ly_vocab)

        self.lx = [x.strip().split() for x in open(lx_corpus).read().split('\n')]
        self.ly = [y.strip().split() for y in open(ly_corpus).read().split('\n')]

        # minus 2 for start/end tokens
        # self.valid_indices = [i for i in range(len(self.lx)) if\
        #                         len(self.lx[i]) < self.max_source_len - 2 and\
        #                         len(self.ly[i]) < self.max_target_len - 2]

        # self.indices = self.make_splits(self.valid_indices)


    def reconstruct_target(self, y_indices):
        """ reconstructs a sentance in the target language  
            TODO - UNK REPLACEMENT FROM ATTENTION
        """
        if y_indices[0] == self.ly_w_to_i[Constants.END]:
            y_indices = y_indices[1:]
        out = []
        for yi in y_indices:
            if yi == self.ly_w_to_i[Constants.START]:
                break
            out.append(self.ly_i_to_w[yi])

        return out


    def reconstruct_source(self, x_indices):
        """ reconstructs a sentence in the target language
        """ 
        if x_indices[0] == self.lx_w_to_i[Constants.END]:  # input was reversed
            x_indices = x_indices[1:]
        out = []
        for yi in x_indices:
            if yi == self.lx_w_to_i[Constants.START]:
                break
            out.append(self.lx_i_to_w[yi])
        return out

        return [self.lx_i_to_w[xi] for xi in x_indices]

    def parse_vocab(self, vocab_file):
        w_to_idx = {}
        for i, w in enumerate(open(vocab_file)):
            w_to_idx[w.strip()] = i
        idx_to_w = {i: x for (x, i) in w_to_idx.iteritems()}
        return w_to_idx, idx_to_w


    def make_splits(self, indices):
        N = len(indices)

        train_test_n = N / 25

        train = indices[:N - (train_test_n * 2)]
        val = indices[len(train): N - train_test_n]
        test = indices[len(train) + len(val):]

        return {'train': train, 'val': val, 'test': test}


    def num_batches(self, dataset='train'):
        return len(self.indices[dataset]) / self.batch_size

    def get_size(self, dataset='train'):
        return len(self.indices[dataset])

    def batch_iter(self, dataset='train'):
        indices = self.indices[dataset]

        while self.has_next_batch(indices):
            yield self.get_batch(indices, self.batch_index)
            self.batch_index += self.batch_size

        self.batch_index = 0


    def has_next_batch(self, indices):
        return self.batch_index + self.batch_size < len(indices)


    def get_batch(self, indices, batch_index):
        def pre_pad(x, pad=self.lx_w_to_i[Constants.PAD]):
            new = [pad] * self.max_source_len
            new[(self.max_source_len - len(x)):] = x
            return new

        def post_pad(y, pad=self.ly_w_to_i[Constants.PAD]):
            new = [pad] * self.max_target_len
            new[:len(y)] = y
            return new

        unk = self.lx_w_to_i[Constants.UNK]
        start = self.lx_w_to_i[Constants.START]
        end = self.lx_w_to_i[Constants.END]
        x_batch = np.array(self.lx)[indices[batch_index: batch_index + self.batch_size]].tolist()
        x_batch = [
            pre_pad(([start] + [self.lx_w_to_i.get(xi, unk) for xi in x] + [end])[::-1]) \
                for x in x_batch
            ]
        x_lens = np.count_nonzero(np.array(x_batch), axis=1).tolist()

        unk = self.ly_w_to_i[Constants.UNK]
        start = self.ly_w_to_i[Constants.START]
        end = self.ly_w_to_i[Constants.END]
        y_batch = np.array(self.ly)[indices[batch_index: batch_index + self.batch_size]].tolist()
        y_batch = [
            post_pad(([start] + [self.ly_w_to_i.get(yi, unk) for yi in y] + [end])[::-1]) \
                for y in y_batch
            ]
        y_lens = np.count_nonzero(np.array(y_batch), axis=1).tolist()

        return x_batch, x_lens, y_batch, y_lens



