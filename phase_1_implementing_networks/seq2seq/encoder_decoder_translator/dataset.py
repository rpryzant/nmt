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
            self.l1_raw, self.l1_dictionary, self.rev_l1_dictionary, self.l1_indices = self.parse_source(l1_path)
            self.l2_raw, self.l2_dictionary, self.rev_l2_dictionary, self.l2_indices = self.parse_source(l2_path)
        else:
            # preprocessed corpus data
            path, self.l1_name, self.l2_name = args
            self.l1_raw, self.l1_dictionary, self.l1_rev_dictionary, self.l1_indices = self.read(path, self.l1_name)
            self.l2_raw, self.l2_dictionary, self.l2_rev_dictionray, self.l2_indices = self.read(path, self.l2_name)

        self.max_seq_len = MAX_SEQ_LEN


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


    def has_next_batch(self, batch_size):
        """ tests whether dataset can emit another batch
        """
        return self.batch_index + batch_size < len(self.l1_indices)


    def reset(self):
        """ resets batch counter to 0
        """
        self.batch_index = 0


    def reconstruct(self, seq, language):
        """ rebuilds the textual representation of a index sequence.
            the language param is used to determine which dictionary to use
               during reconstruction
        """
        if language == self.l1_name:
            return ' '.join(self.l1_rev_dictionary[x] for x in seq)
        elif language == self.l2_name:
            return ' '.join(self.l2_rev_dictionary[x] for x in seq)
        print 'ERROR: language %s unrecognized! Supported languages are %s and %s.' % \
            (language, self.l1_name, self.l2_name)


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
                ln = ln
            return seq, ln

        x = self.l1_indices[self.batch_index : self.batch_index + batch_size]
        y = self.l2_indices[self.batch_index : self.batch_index + batch_size]

        self.batch_index += batch_size

        l = []
        for i in range(batch_size):
            lengths = [-1, -1]
            x[i], lengths[0] = clip_pad(x[i], self.l1_dictionary)
            y[i], lengths[1] = clip_pad(y[i], self.l2_dictionary)
            l.append(lengths)

        return x, y, l


        


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




