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
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 50

class Dataset:

    def __init__(self, *args):
        self.batch_index = 0
        if len(args) == 2:
            # raw corpus data
            l1_path, l2_path = args
            self.l1_name, self.l2_name = 'l1', 'l2'
            self.l1_raw, self.l1_dictionary, self.l1_rev_dictionary, self.l1_indices = self.parse_source(l1_path)
            self.l2_raw, self.l2_dictionary, self.l2_rev_dictionray, self.l2_indices = self.parse_source(l2_path)
        else:
            # preprocessed corpus data
            path, self.l1_name, self.l2_name = args
            self.l1_raw, self.l1_dictionary, self.l1_rev_dictionary, self.l1_indices = self.read(path, self.l1_name)
            self.l2_raw, self.l2_dictionary, self.l2_rev_dictionary, self.l2_indices = self.read(path, self.l2_name)

        self.max_seq_len = MAX_SEQ_LEN
        self.backups = {
                'l1_raw': self.l1_raw, 
                'l1_raw': self.l1_dictionary, 
                'l1_rev_dict': self.l1_rev_dictionary,
                'l1_indices': self.l1_indices,
                'l2_raw': self.l2_raw, 
                'l2_raw': self.l2_dictionary, 
                'l2_rev_dict': self.l2_rev_dictionary,
                'l2_indices': self.l2_indices}


    def read(self, path, language):
        """ reads sentances, dictionary, and sentence indices from preprocessed corpus
        """
        base = '%s/%s.' % (path, language)
        raw = np.load(base + 'raw.npy')
        dictionary = np.load(base + 'dictionary.npy')
        dictionary = dict([(x[0], int(x[1])) for x in dictionary ])
        rev_dictionary = {v: k for (k, v) in dictionary.iteritems()}
        indices = np.load(base + 'indices.npy')

        return raw, dictionary, rev_dictionary, indices


    def write(self, path, l1_name, l2_name):
        """ write (1) raw sentances, (2) vocab dictionary, and (3) indexed sentances
            for both languages to the location specified by path

            The "lX_name" should be the name of each language, e.g. "en" and "vi".
             we need this param because datasets are language-invarient 
        """
        l1_base = '%s/%s.' % (path, l1_name)
        np.save(l1_base + 'raw', np.array(self.l1_raw))
        np.save(l1_base + 'dictionary', np.array(self.l1_dictionary.items()))
        np.save(l1_base + 'indices', np.array(self.l1_indices))

        l2_base = '%s/%s.' % (path, l2_name)
        np.save(l2_base + 'raw', np.array(self.l2_raw))
        np.save(l2_base + 'dictionary', np.array(self.l2_dictionary.items()))
        np.save(l2_base + 'indices', np.array(self.l2_indices))


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
        vocab = f.most_common(VOCAB_SIZE - 1)

        reverse_index = {w: i+1 for i, (w, f) in enumerate(vocab)}
        reverse_index[UNK] = VOCAB_SIZE
        index = {v: k for (k, v) in reverse_index.iteritems()}


        src_tok_unk = [[w if w in reverse_index else UNK for w in s] for s in src_tok]
        treated_corpus = [[reverse_index[w] for w in sent] for sent in src_tok_unk]

        return src_tok, reverse_index, index, treated_corpus


    def set_language_names(self, n1, n2):
        self.l1_name = n1
        self.l2_name = n2


    def subset(self, N):
        """ subset internal data for faster training
        """
        self.l1_raw = self.l1_raw[:N]
        self.l1_indices = self.l1_indices[:N]
        self.l2_raw = self.l2_raw[:N]
        self.l2_indices = self.l2_indices[:N]


    def reset(self):
        """ resets batch counter to 0
        """
        self.batch_index = 0


    def cancel_subset(self):
        """ resets batch counter to 0 and restores from subset
        """
        self.batch_index = 0
        self.l1_raw = self.backups['l1_raw']
        self.l1_indices = self.backups['l1_indices']
        self.l2_raw = self.backups['l2_raw']
        self.l2_indices = self.backups['l2_indices']


    def reconstruct(self, seq, language):
        """ rebuilds the textual representation of a index sequence.
            the language param is used to determine which dictionary to use
               during reconstruction
        """
        if language == self.l1_name:
            return ' '.join(self.l1_rev_dictionary.get(x, '') for x in seq)
        elif language == self.l2_name:
            return ' '.join(self.l2_rev_dictionary.get(x, '') for x in seq)
        print 'ERROR: language %s unrecognized! Supported languages are %s and %s.' % \
            (language, self.l1_name, self.l2_name)


    def has_next_batch(self, batch_size):
        """ tests whether dataset can emit another batch
        """
        return self.batch_index + batch_size < len(self.l1_indices)


    def next_batch(self, batch_size):
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


        l = []
        x = self.l1_indices[self.batch_index : self.batch_index + batch_size].tolist()
        y = self.l2_indices[self.batch_index : self.batch_index + batch_size].tolist()

        self.batch_index += batch_size

        x_batch = [clip_pad(x[i], self.l1_dictionary)[0] for i in range(batch_size)]
        x_lens = np.count_nonzero(np.array(x_batch), axis=1).tolist()

        y_batch = [clip_pad(y[i], self.l2_dictionary)[0] for i in range(batch_size)]
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




