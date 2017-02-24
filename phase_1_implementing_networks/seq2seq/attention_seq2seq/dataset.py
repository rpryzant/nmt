"""
=== DESCRIPTION
Class for managing en-alt parallel corpora and performing all 
  corpus-related operations 

"alt" stands for "alternative" corpus, i.e. non-english 
  (so western-centric sigh)

=== USAGE
= from plaintext corpus
python dataset.py data/raw/train.en data/raw/train.vi  
= from preprocessed corpus
python dataset.py data/processed/ en vi
"""
import sys
import nltk
import itertools
import numpy as np
import string


START = "_START_"
END = "_END_"
UNK = "_UNK_"

class Dataset:

    def __init__(self, config, *args):
        self.config = config
        self.vocab_size = config.src_vocab_size - 1  # config includes a +1 for unks
        self.max_seq_len = config.max_source_len
        self.batch_size = self.config.batch_size

        self.batch_index = 0
        if len(args) == 2:
            # raw corpus data
            l1_path, l2_path = args
            self.l1_name, self.l2_name = 'l1', 'l2'
            self.l1_raw, self.l1_word_index, self.l1_rev_word_index, self.l1_sentences = self.parse_source(l1_path)
            self.l2_raw, self.l2_word_index, self.l2_rev_word_index, self.l2_sentences = self.parse_source(l2_path)
        else:
            # preprocessed corpus data
            path, self.l1_name, self.l2_name = args
            self.l1_raw, self.l1_word_index, self.l1_rev_word_index, self.l1_sentences = self.read(path, self.l1_name)
            self.l2_raw, self.l2_word_index, self.l2_rev_word_index, self.l2_sentences = self.read(path, self.l2_name)

        self.make_train_test_splits()

        self.backups = {
                'l1_raw': self.l1_raw, 
                'l1_raw': self.l1_word_index, 
                'l1_rev_dict': self.l1_rev_word_index,
                'l1_sentences': self.l1_sentences,
                'l2_raw': self.l2_raw, 
                'l2_raw': self.l2_word_index, 
                'l2_rev_dict': self.l2_rev_word_index,
                'l2_sentences': self.l2_sentences}


    def read(self, path, language):
        """ reads sentances, dictionary, and sentence indices from preprocessed corpus
        """
        base = '%s/%s.' % (path, language)
        raw = np.load(base + 'raw.npy')
        index = np.load(base + 'dictionary.npy')
        index = dict([(x[0], int(x[1])) for x in index ])
        rev_index = {v: k for (k, v) in index.iteritems()}
        indices = np.load(base + 'indices.npy')

        return raw, index, rev_index, indices


    def write(self, path, l1_name, l2_name):
        """ write (1) raw sentances, (2) vocab dictionary, and (3) indexed sentances
            for both languages to the location specified by path

            The "lX_name" should be the name of each language, e.g. "en" and "vi".
             we need this param because datasets are language-invarient 
        """
        l1_base = '%s/%s.' % (path, l1_name)
        np.save(l1_base + 'raw', np.array(self.l1_raw))
        np.save(l1_base + 'dictionary', np.array(self.l1_word_index.items()))
        np.save(l1_base + 'indices', np.array(self.l1_sentences))

        l2_base = '%s/%s.' % (path, l2_name)
        np.save(l2_base + 'raw', np.array(self.l2_raw))
        np.save(l2_base + 'dictionary', np.array(self.l2_word_index.items()))
        np.save(l2_base + 'indices', np.array(self.l2_sentences))


    def parse_source(self, path):
        """ parses a space-seperated corpus into
              1) a list of constituent sentances with start & end tokens added
              2) a dictionary mapping the top K most frequent words to their rank indices
              3) a list of sentances with words converted to their index 
        
            note that 0 is a special index reserved for for unknown/out-of-dictionary words
        """
        f = open(path, 'rb')
        src = [s.translate(None, string.punctuation) for s in f]
        src = [s.strip().decode('utf-8').lower() for s in src]
        src = ['%s %s %s' % (START, x, END) for x in src]
        src_tok = [nltk.word_tokenize(x) for x in src]

        f = nltk.FreqDist(itertools.chain(*src_tok))
        vocab = f.most_common(self.vocab_size - 1)

        word_index = {w: i+1 for i, (w, f) in enumerate(vocab)}
        word_index[UNK] = self.vocab_size
        reverse_index = {v: k for (k, v) in word_index.iteritems()}

        src_tok_unk = [[w if w in word_index else UNK for w in s] for s in src_tok]
        treated_corpus = [[word_index[w] for w in sent] for sent in src_tok_unk]

        return src_tok, word_index, reverse_index, treated_corpus


    def set_language_names(self, n1, n2):
        self.l1_name = n1
        self.l2_name = n2


    def subset(self, N_new):
        """ subset internal data for faster training
        """
        self.l1_raw = self.l1_raw[:N_new]
        self.l1_sentences = self.l1_sentences[:N_new]
        self.l2_raw = self.l2_raw[:N_new]
        self.l2_sentences = self.l2_sentences[:N_new]
        self.make_train_test_splits()


    def make_train_test_splits(self):
        self.N = len(self.l1_sentences)
        self.train_indices = range(self.N - (self.N/10))
        self.train_N = len(self.train_indices)
        self.val_indices = range(self.train_indices[-1], self.N)
        self.val_N = len(self.val_indices)


    def reset_batch_counter(self):
        """ resets batch counter to 0
        """
        self.batch_index = 0


    def cancel_subset(self):
        """ resets batch counter to 0 and restores from subset
        """
        self.batch_index = 0
        self.l1_raw = self.backups['l1_raw']
        self.l1_sentences = self.backups['l1_sentences']
        self.l2_raw = self.backups['l2_raw']
        self.l2_sentences = self.backups['l2_sentences']
        self.make_train_test_splits()

    def get_start_token_indices(self):
        """ start token index for both languages
        """
        return self.l1_word_index[START], self.l2_word_index[START]

    def get_end_token_indices(self):
        return self.l1_word_index[END], self.l2_word_index[END]


    def reconstruct(self, seq, language):
        """ rebuilds the textual representation of a index sequence.
            the language param is used to determine which dictionary to use
               during reconstruction
        """
        if language == self.l1_name:
            return ' '.join(self.l1_rev_word_index.get(x, '') for x in seq)
        elif language == self.l2_name:
            return ' '.join(self.l2_rev_word_index.get(x, '') for x in seq)
        print 'ERROR: language %s unrecognized! Supported languages are %s and %s.' % \
            (language, self.l1_name, self.l2_name)


    def batch_iter(self, training=True):
        while self.has_next_batch(training):
            yield self.next_batch(training)


    def has_next_batch(self, training=True):
        """ tests whether dataset can emit another batch
        """
        data_N = self.train_N if training else self.val_N
        return self.batch_index + self.batch_size < data_N


    def next_batch(self, training=True):
        """ returns next batch_size examples and their lengths
            pads, clips, and measures lengths
        """
        def clip_pad(seq, dict):
            ln = len(seq)
            if ln > self.max_seq_len:
                seq = seq[:self.max_seq_len-1]
                seq += [dict[END]]
                ln = self.max_seq_len
            else:
                seq += [0] * (self.max_seq_len - ln)
            return seq, ln

        index_source = self.train_indices if training else self.val_indices
        indices = index_source[self.batch_index : self.batch_index + self.batch_size]
        x = self.l1_sentences[indices].tolist()
        y = self.l2_sentences[indices].tolist()

        self.batch_index += self.batch_size

        x_batch = [clip_pad(x[i], self.l1_word_index)[0] for i in range(self.batch_size)]
        x_lens = np.count_nonzero(np.array(x_batch), axis=1).tolist()

        y_batch = [clip_pad(y[i], self.l2_word_index)[0] for i in range(self.batch_size)]
        y_lens = np.count_nonzero(np.array(y_batch), axis=1).tolist()

        return x_batch, x_lens, y_batch, y_lens


        


if __name__ == "__main__":
    d = Dataset(*tuple(sys.argv[1:]))
    d.write('./data/processed', 'en', 'vi')


#    while d.has_next_batch(3):
#        x, y, l = d.next_batch(3)
#        print x[0]
#        print y[0]
#        print l[0]
#        print 
#        print




