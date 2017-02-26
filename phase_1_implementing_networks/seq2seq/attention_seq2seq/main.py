"""


=== DESCRIPTION
This is a basic seq2seq translation system without any 
bells or whistles. The model is in good shape, but this main
file is pretty horrid I admit. That's fine, though. If you run
it as-is, the model will overfit on a subset of the provided data. 


=== USAGE
python main.py data/processed_30/ en vi dot 50 [checkpoint path to load from]

"""
import numpy as np
from dataset import Dataset
from model import Seq2SeqV3, AttentionNN
import sys
import utils
import time
from tqdm import tqdm
import os
import utils




data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]
attention_type = sys.argv[4]
epochs = int(sys.argv[5])
model_path = sys.argv[6] if len(sys.argv) > 6 else None

c = utils.Config()
c.attention = attention_type

print 'INFO: building dataset...'
d = Dataset(c, data_loc, lang1, lang2)
print 'INFO: dataset built. Train size: ', d.train_N, 'val size: ', d.val_N
#d.subset(13)    # take only X sentances
#print 'INFO: subset built. Train size: ', d.train_N, 'val size: ', d.val_N



print 'INFO: building model...'
#model = Seq2SeqV3(c, d, testing=False)
model = AttentionNN(c, d, testing=False)
if model_path is not None:
    model.load(model_path)
print 'INFO: model built.'


print 'INFO: building checkpoint dir...'
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if not os.path.exists(cur_dir + c.checkpoint_dir):
    os.mkdir(cur_dir + c.checkpoint_dir)
checkpoint_dir = cur_dir + c.checkpoint_dir + '/%s_%s_%s' % (lang1, lang2, attention_type)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
print 'INFO: checkpoints ready to go'


print 'INFO: training...'
lr = c.learning_rate
best_valid_loss = float("inf")
train_losses = []
val_losses = []
try:
    for epoch in range(epochs):
        start = time.time()

        # training
        i = 0
        train_loss = 0.0
        for batch in tqdm(d.batch_iter(training=True), total=d.train_N/c.batch_size):
#        for batch in d.batch_iter(training=True):
            pred, loss, _ = model.train_on_batch(*batch, learning_rate=lr)
            i += 1.0
            train_loss += loss
        train_loss /= (i or 1)
        train_losses.append(train_loss)
        d.reset_batch_counter()
        # validation
        i = 0.0
        val_loss = 0.0
        for batch in tqdm(d.batch_iter(training=False), total=d.val_N/c.batch_size):
#        for batch in d.batch_iter(training=False):
            pred, loss = model.run_on_batch(*batch, learning_rate=lr)
            val_loss += loss
            i += 1.0
        val_loss /= (i or 1)
        val_losses.append(val_loss)
        d.reset_batch_counter()

        if epoch > 0 and val_loss < best_valid_loss:
            print 'INFO: new best validation loss!...'
            best_valid_loss = val_loss
            print 'INFO: generating plots...'
            filename = checkpoint_dir + '/%s-%s.png' % (attention_type, str(epoch))
            utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                            [(train_losses, 'train'), (val_losses, 'val')])
            print 'INFO: plot saved to %s' % filename

            print 'INFO: saving checkpoint...'
            checkpoint_loc = checkpoint_dir + '/checkpoint-%s' % epoch
            model.save(checkpoint_loc)
            print 'INFO: checkpoint saved to %s' % (checkpoint_loc)

        if epoch > 9 and epoch % 5 == 0 and lr > 0.0025:
            lr = lr * 0.75

        print 'INFO: epoch,', epoch, 'train loss,', train_loss, 'val loss,', val_loss, 'time, ', (time.time() - start)

except KeyboardInterrupt:
    print 'INFO: stopped!'

finally:
    print 'INFO: generating plots...'
    filename = checkpoint_dir + '/final_losses.png'
    utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                    [(train_losses, 'train'), (val_losses, 'val')])
    print 'INFO: plot saved to %s' % filename


quit()
