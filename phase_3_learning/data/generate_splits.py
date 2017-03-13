"""

=== DESCRIPTION
Given two corpus files that contain corresponding translations,
  one per line, this script generates train/dev/test indices 

=== USAGE
python generate_splits.py [en file] [other file]

"""
import sys; sys.path.append('../')    # sigh 
from msc.utils import file_length
import random


max_source_len = 50
max_target_len = 50
N = 1600000
train_test_n = N / 50
train_n = N - (train_test_n * 2)

corpus = sys.argv[1]


# read in corpii (?)
lx = [x.strip().split() for x in open(sys.argv[1]).read().split('\n')]
ly = [y.strip().split() for y in open(sys.argv[2]).read().split('\n')]


# get all sentences that are short enough for consideration
#    (minus 2 for start/end tokens)
valid_indices = [i for i in range(len(lx)) if\
                        len(lx[i]) < max_source_len - 2 and\
                        len(ly[i]) < max_target_len - 2]


# shuffle indices and select train/test/val splits
random.shuffle(valid_indices)

indices = valid_indices[:N]
train = indices[:train_n]
val = indices[train_n: train_n + train_test_n]

test = indices[-train_test_n:]


# write output
f= open('train', 'w')
f.write('\n'.join(str(x) for x in train))
f.close()

f= open('val', 'w')
f.write('\n'.join(str(x) for x in val))
f.close()

f= open('test', 'w')
f.write('\n'.join(str(x) for x in test))
f.close()



