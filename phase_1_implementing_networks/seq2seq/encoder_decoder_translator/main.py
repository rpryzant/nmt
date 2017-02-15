"""
python main.py data/processed/ en vi
"""

from dataset import Dataset
from model import Seq2Seq
import sys



class config:
    src_vocab_size = 5000
    max_source_len = 50
    embedding_size = 64
    hidden_size = 128
    dropout_rate = 0.5
    num_layers = 3
    tgt_vocab_size = 5000
    max_target_len = 50
    learning_rate = 0.0003


data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]

batch_size = 5
print 'building dataset...'
d = Dataset(data_loc, lang1, lang2)
d.subset(2000)    # take only 2k sentances
print 'building model...'
c = config()
model = Seq2Seq(c, batch_size)

print 'training...'
for epoch in range(100):
    epoch_loss = 0.0
    i = 0
    while d.has_next_batch(batch_size):
        [x, y, l] = d.next_batch(batch_size)
        print x
        print
        print y
        print
        print l
        _, loss = model.train_on_batch(x, y, l)
        epoch_loss += loss
        i += 1
    epoch_loss /= (i * batch_size)
    print 'epoch=%s\t mean batch loss=%d' % (epoch, epoch_loss)
