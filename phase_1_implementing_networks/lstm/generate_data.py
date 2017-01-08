import sys
import itertools
import nltk
import h5py
import numpy as np

VOCAB_SIZE = 8000
UNK = "_UNK"
SENTENCE_START = "_START"
SENTENCE_END = "_END"

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

# build word to index mapping
word_to_index = {w: i for i, (w, f) in enumerate(vocabulary)}

# replace words not in vocab with unk
for i, s in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else UNK for w in s]

# create and save datasets
X = [[word_to_index[w] for w in s[:-1]] for s in tokenized_sentences] # exclude _END
Y = [[word_to_index[w] for w in s[1:]] for s in tokenized_sentences] # exclude _START

print [X, Y]
