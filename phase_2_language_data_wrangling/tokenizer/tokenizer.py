
"""
=== DESCRIPTION
This script takes an ASPEC corpus file as input, and produces 
  tokenized monolingual corpus files as output

=== USAGE
python tokenizer.py [en corpusfile] [ja corpusfile]
"""
from rakutenma import RakutenMA
import sys
import os
import codecs
import json
from joblib import Parallel, delayed






model = 'rakuten_model_ja.min.json' 

def process(f):
    ja_file = f
    en_file = f.replace('ja', 'en')
    print 'WORKING ON'
    print ja_file
    print en_file

    suffix = f.split('.')[-1]
    input = open(f)

    # bulid tokenizers
    ja_tokenizer = RakutenMA(json.loads(open(model).read()))
    ja_tokenizer.hash_func = RakutenMA.create_hash_func(ja_tokenizer, 15)

    en_outfile = codecs.open('tokenized/en_tmp.' + suffix, 'w', 'utf-8') 
    ja_outfile = codecs.open('tokenized/ja.' + suffix, 'w', 'utf-8')

    # parse en/ja lines, tokenize ja with rakuten
    en_lines = []
    ja_lines = []
    for i, (en, ja) in enumerate(zip(open(en_file), open(ja_file))):
        if i % 10000 == 0:  print i

        en_outfile.write(' '.join(x for x in en.lower().strip().decode('utf-8').split()) + '\n')
        ja_outfile.write(' '.join([x[0] for x in ja_tokenizer.tokenize(ja.strip().decode('utf-8'))]) + '\n')

    en_outfile.close()
    ja_outfile.close()

    # tokenize en with moses
    os.system('perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 < %s > %s' % ('tokenized/en_tmp.' + suffix, 'tokenized/en.' + suffix))




en_corpus = sys.argv[1]
ja_corpus = sys.argv[2]


print 'preparing run...'
os.system('rm -r split_files; mkdir split_files')
os.system('rm -r tokenized; mkdir tokenized')
os.system('split -l 25000 %s split_files/train.en.' % en_corpus)
os.system('split -l 25000 %s split_files/train.ja.' % ja_corpus)
print 'starting processing!'
Parallel(n_jobs=16)(delayed(process)(os.path.join('split_files/', f)) \
                       for f in os.listdir('split_files/') if f.startswith('train.ja'))

print 'done! joining...'
split_order = []
for f in os.listdir('tokenized'):

    if 'ja' in f:
        split_order.append(f.split('.')[1])

ja_cat = 'cat ' + ' '.join('tokenized/ja.%s' % split for split in split_order) + ' > tokenized/ja.tok'
en_cat = 'cat ' + ' '.join('tokenized/en.%s' % split for split in split_order) + ' > tokenized/en.tok'


os.system(ja_cat)
os.system(en_cat)


