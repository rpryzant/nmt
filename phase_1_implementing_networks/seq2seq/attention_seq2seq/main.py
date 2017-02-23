"""


=== DESCRIPTION
This is a basic seq2seq translation system without any 
bells or whistles. The model is in good shape, but this main
file is pretty horrid I admit. That's fine, though. If you run
it as-is, the model will overfit on a subset of the provided data. 


=== USAGE
python main.py data/processed/ en vi tmp/checkpoint.ckpt-103 load

"""
import numpy as np
from dataset import Dataset
from model import Seq2SeqV3
import sys
import utils

class config:
    src_vocab_size = 5000 + 1 # +1 for unk
    max_source_len = 50
    embedding_size = 64
    batch_size = 5
    hidden_size = 128
    dropout_rate = 0.5
    num_layers = 3
    target_vocab_size = 5000 + 1 # +1 for unk
    max_target_len = 50
    learning_rate = 1.0    # sgd
    attention = 'off'       # accepted values: [off, dot, bilinear]
#    learning_rate = 0.001  # adam


data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]
model_path = sys.argv[4]
restore = sys.argv[5] if len(sys.argv) == 6 else None

c = config()

print 'INFO: building dataset...'
d = Dataset(c, data_loc, lang1, lang2)
print 'INFO: dataset built.'
#d.subset(6)    # take only 6 sentances


print 'INFO: building model...'
model = Seq2SeqV3(c, d, testing=False)
print 'INFO: model built.'



print 'INFO: training...'
lr = c.learning_rate

train_losses = []
val_losses = []
try:
    for epoch in range(20000):
        # training
        i = 0
        train_loss = 0.0
        while d.has_next_batch(training=True):
            i += 1
            batch = d.next_batch(training=True)
            pred, loss = model.train_on_batch(*batch, learning_rate=lr)
            train_loss += loss
        train_losses.append(train_loss / i)
        d.reset_batch_counter()

        # validation
        i = 0.0
        val_loss = 0.0
        while d.has_next_batch(training=False):
            batch = d.next_batch(training=False)
            pred, loss = model.run_on_batch(*batch, learning_rate=lr)
            val_loss += loss
        val_losses.append(val_loss / i)
        d.reset_batch_counter()

except KeyboardInterrupt:
    print 'INFO: stopped!'

finally:
    print 'INFO: generating plots...'
    filename = 'losses.png'
    utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                    [(train_losses, 'train'), (val_losses, 'val')])
    print 'INFO: plot saved to %s' % filename



