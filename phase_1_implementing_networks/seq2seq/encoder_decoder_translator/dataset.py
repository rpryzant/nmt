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

"""

input_data = sys.argv[1]

f = open(input_data, 'rb')

# read in data and append sentance boundary tokens                                                                                    
raw_text = ' '.join(line.strip() for line in f if len(line) > 0)
sentences = itertools.chain(nltk.sent_tokenize(raw_text.decode('utf-8').lower()))
sentences = ['%s %s %s' % (SENTENCE_START, s, SENTENCE_END) for s in sentences]

# tokenize into words                                                                                                                 
tokenized_sentences = [nltk.word_tokenize(s) for s in sentences]

# count word freqs                                                                                                                    
freqs = nltk.FreqDist(itertools.chain(*tokenized_sentences))

# get most common words: [(word, freq)] sorted by freq                                                                                
vocabulary = freqs.most_common(VOCAB_SIZE - 1)

# build word to index mapping, and +1 to all id's to reserve 0 for padding                                                            
word_to_index = {w: i+1 for i, (w, f) in enumerate(vocabulary)}

# replace words not in vocab with unk                                                                                                 
for i, s in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else UNK for w in s]

# create and save datasets                                                                                                            
X = [[word_to_index[w] for w in s[:-1]] for s in tokenized_sentences] # exclude _END                                                  
Y = [[word_to_index[w] for w in s[1:]] for s in tokenized_sentences] # exclude _START                                                 
"""

START = "<s>"
END = "</s>"

class Dataset:

    def __init__(self, en_path, alt_path):
        f = open(en_path, 'rb')
        s = [s.strip().decode('utf-8').lower() for s in f]

        # TODO HERE^^^^
        # IGNORE WHATS BELOW THIS




        print s
        s = itertools.chain(nltk.sent_tokenize(nltk.sent_tokenize(text.decode('utf-8').lower())))
        print s
        quit()


        self.en_path = en_path
        self.en_s_index, self.en_rev_s_index =  self.__build_sentence_indices(en_path)
        self.N_en_s = len(self.en_s_index)

        self.alt_path = alt_path
        self.alt_s_index, self.alt_rev_s_index = self.__build_sentence_indices(alt_path)
        self.N_alt_s = len(self.alt_s_index)




    def __build_sentence_indices(self, path):
        index     = { s.strip(): i for i, s in enumerate(open(path).read().splitlines()) }
        rev_index = { i: s.strip() for i, s in enumerate(open(path).read().splitlines()) }
        return index, rev_index


if __name__ == "__main__":
    d = Dataset(sys.argv[1], sys.argv[2])
