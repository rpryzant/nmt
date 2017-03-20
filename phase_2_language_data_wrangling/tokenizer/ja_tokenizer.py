
"""
=== DESCRIPTION
This script takes some japanese phrases and tokenizes them

=== USAGE
python tokenizer.py [ja corpusfile]
"""
from rakutenma import RakutenMA
import sys
import os
import codecs
import json
from joblib import Parallel, delayed






model = 'rakuten_model_ja.min.json' 

ja_corpus = open(sys.argv[1])

ja_tokenizer = RakutenMA(json.loads(open(model).read()))
ja_tokenizer.hash_func = RakutenMA.create_hash_func(ja_tokenizer, 15)

for ja in ja_corpus:
    print ' '.join([x[0].encode('utf-8') for x in ja_tokenizer.tokenize(ja.decode('utf-8').strip())])
