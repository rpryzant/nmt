"""
=== DESCRIPTION
Class for managing en-alt parallel corpora and performing all 
  corpus-related operations 

"alt" stands for "alternative" corpus, i.e. non-english 
  (so western-centric sigh)

=== USAGE
= from plaintext corpus
python dataset.py data/train.en data/train.vi  
= from preprocessed corpus
python dataset.py ex_parsed_data/ en vi
"""
import sys
import nltk
import itertools
import numpy as np



START = "_START_"
END = "_END_"
UNK = "_UNK_"
VOCAB_SIZE = 5000


class Dataset:

    def __init__(self, l1_path, l2_path):
        """ constructor for raw corpus data
        """
        self.l1_raw, self.l1_dictionary, self.l1_indices = self.parse_source(l1_path)
        self.l2_raw, self.l2_dictionary, self.l2_indices = self.parse_source(l2_path)


    def __init__(self, path, l1_name, l2_name):
        """ constructor for pre-processed corpus data
        """
        self.l1_raw, self.l1_dictionary, self.l1_indices = self.read(path, l1_name)
        self.l2_raw, self.l2_dictionary, self.l2_indices = self.read(path, l2_name)
        print self.l2_indices


    def read(self, path, language):
        """ reads sentances, dictionary, and sentence indices from preprocessed corpus
        """
        base = '%s/%s.' % (path, language)
        raw = np.load(base + 'dictionary.npy')
        dictionary = np.load(base + 'dictionary.npy')
        dictionary = dict([(x[0], int(x[1])) for x in dictionary ])
        indices = np.load(base + 'indices.npy')

        return raw, dictionary, indices


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
        s = [s.strip().decode('utf-8').lower() for s in f]
        s = ['%s %s %s' % (START, x, END) for x in s]

        tok_s = [nltk.word_tokenize(x) for x in s]
        f = nltk.FreqDist(itertools.chain(*tok_s))
        vocab = f.most_common(VOCAB_SIZE - 1)

        reverse_index = {w: i+1 for i, (w, f) in enumerate(vocab)}
        reverse_index[UNK] = 0

        for i, s in enumerate(tok_s):
            tok_s[i] = [w if w in reverse_index else UNK for w in s]

        treated_corpus = [[reverse_index[w] for w in sent] for sent in tok_s]

        return s, reverse_index, treated_corpus




if __name__ == "__main__":
#    d = Dataset(sys.argv[1], sys.argv[2])
#    d.write('./test_out', 'en', 'vi')

    d = Dataset(sys.argv[1], sys.argv[2], sys.argv[3])
