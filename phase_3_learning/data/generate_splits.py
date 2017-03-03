

import sys; sys.path.append('../')    # sigh 
from msc.utils import file_length
import random


max_source_len = 50
max_target_len = 50


corpus = sys.argv[1]

lx = [x.strip().split() for x in open(sys.argv[1]).read().split('\n')]
ly = [y.strip().split() for y in open(sys.argv[2]).read().split('\n')]


# minus 2 for start/end tokens
valid_indices = [i for i in range(len(lx)) if\
                        len(lx[i]) < max_source_len - 2 and\
                        len(ly[i]) < max_target_len - 2]






random.shuffle(valid_indices)

indices = valid_indices[:110000]
N = len(indices)
train_test_n = N / 25


train = indices[:N - (train_test_n * 2)]
val = indices[len(train): N - train_test_n]
test = indices[len(train) + len(val):]


f= open('train', 'w')
f.write('\n'.join(str(x) for x in train))
f.close()

f= open('val', 'w')
f.write('\n'.join(str(x) for x in val))
f.close()

f= open('test', 'w')
f.write('\n'.join(str(x) for x in test))
f.close()



