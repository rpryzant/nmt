"""
=== DESCRIPTION
Class for managing en-alt parallel corpora and performing all 
  corpus-related operations 

"alt" stands for "alternative" corpus, i.e. non-english 
  (so western-centric sigh)

=== USAGE
python dataset.py data/train.en data/train.alt  


"""
import sys
import nltk
import itertools




START = "_START_"
END = "_END_"
UNK = "_UNK_"
VOCAB_SIZE = 5000


class Dataset:

    def __init__(self, l1_path, l2_path):
        self.l1_raw, self.l1_dictionary, self.l1_indices = self.parse_source(l1_path)
        self.l2_raw, self.l2_dictionary, self.l2_indices = self.parse_source(l2_path)



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
    d = Dataset(sys.argv[1], sys.argv[2])
