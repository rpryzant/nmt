"""
=== DESCRIPTION
This file generates vocabulary files for a given corpus.
Expected input is a list of sentences, one per line, with
  word-tokens seperated by spaces

=== USAGE
python generate_vocabulary.py [corpus] [max_vocab_size] > outfile
"""
from collections import Counter
import sys; sys.path.append('../')    # sigh 
from msc.utils import Constants 


corpus = sys.argv[1]
num_words = int(sys.argv[2])

words = open(corpus).read().decode('utf-8').lower().split()
c = Counter(words)

print Constants.PAD
print Constants.START
print Constants.END
print Constants.UNK

for w, cnt in c.most_common(num_words):
    print w.encode('utf-8')
