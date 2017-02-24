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
import time
from tqdm import tqdm
import os


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
    attention = 'dot'       # accepted values: [off, dot, bilinear]
#    learning_rate = 0.001  # adam


data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]
attention_type = sys.argv[4]
epochs = int(sys.argv[5])

c = config()
c.attention = attention_type

print 'INFO: building dataset...'
d = Dataset(c, data_loc, lang1, lang2)
print 'INFO: dataset built.'
d.subset(6)    # take only 6 sentances


print 'INFO: building model...'
model = Seq2SeqV3(c, d, testing=False)
print 'INFO: model built.'


print 'INFO: building checkpoint dir...'
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if not os.path.exists(cur_dir + 'checkpoints'):
    os.mkdir(cur_dir + 'checkpoints')
checkpoint_dir = cur_dir + 'checkpoints/%s_%s_%s' % (lang1, lang2, attention_type)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
print 'INFO: checkpoints ready to go'


print 'INFO: training...'
lr = c.learning_rate
train_losses = []
val_losses = []
try:
    for epoch in range(epochs):
        start = time.time()

        # training
        i = 0
        train_loss = 0.0
        for batch in tqdm(d.batch_iter(training=True), total=d.train_N):
            pred, loss = model.train_on_batch(*batch, learning_rate=lr)
            i += 1.0
#            print 'train', i, loss
            train_loss += loss
        train_loss /= (i or 1)
        train_losses.append(train_loss)
        d.reset_batch_counter()

        # validation
        i = 0.0
        val_loss = 0.0
        for batch in tqdm(d.batch_iter(training=False), total=d.val_N):
            pred, loss = model.run_on_batch(*batch, learning_rate=lr)
            val_loss += loss
            i += 1.0
#            print 'val', i, loss
        val_loss /= (i or 1)
        val_losses.append(val_loss)
        d.reset_batch_counter()

        if epoch > 0 and epoch % 10 == 0:
            lr *= 0.95
            print 'INFO: generating plots...'
            filename = checkpoint_dir + '/%s-%s.png' % (attention_type, str(epoch))
            utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                            [(train_losses, 'train'), (val_losses, 'val')])
            print 'INFO: plot saved to %s' % filename

            print 'INFO: saving checkpoint...'
            model.save_checkpoint(checkpoint_dir + '/checkpoint')
            print 'INFO: checkpoint saved to %s' % (checkpoint_dir + '/checkpoint')

        print 'INFO: epoch,', epoch, 'train loss,', train_loss, 'val loss,', val_loss, 'time, ', (time.time() - start)

except KeyboardInterrupt:
    print 'INFO: stopped!'

finally:
    print 'INFO: generating plots...'
    filename = 'final_losses.png'
    utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                    [(train_losses, 'train'), (val_losses, 'val')])
    print 'INFO: plot saved to %s' % filename


quit()
