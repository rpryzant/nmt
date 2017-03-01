"""


=== DESCRIPTION
This is a basic seq2seq translation system without any 
bells or whistles. The model is in good shape, but this main
file is pretty horrid I admit. That's fine, though. If you run
it as-is, the model will overfit on a subset of the provided data. 


=== USAGE
python python main.py datasets/raw/ en vi [checkpoint path to load from]

"""
import numpy as np
from data.data_iterator import Dataset
from models.seq2seq import Seq2SeqV3
import sys
import msc.utils
import time
from tqdm import tqdm
import os
from msc.utils import Config, lineplot, file_length




data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]
model_path = sys.argv[4] if len(sys.argv) > 4 else None


c = Config()
c.src_vocab_size = file_length(data_loc + c.x_vocab)    # hacky...make better
c.target_vocab_size = file_length(data_loc + c.y_vocab)


print 'INFO: building dataset...'
#d = Dataset(c, data_loc, lang1, lang2)
d = Dataset(c, data_loc)
i = 0
print 'INFO: dataset built. Train size: ', d.get_size('train'), 'val size: ', d.get_size('val')
#d.subset(13)    # take only X sentances
#print 'INFO: subset built. Train size: ', d.train_N, 'val size: ', d.val_N



print 'INFO: building model...'
model = Seq2SeqV3(c, d, testing=False)
if model_path is not None:
    model.load(model_path)
print 'INFO: model built.'


print 'INFO: building checkpoint dir...'
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if not os.path.exists(cur_dir + c.checkpoint_dir):
    os.mkdir(cur_dir + c.checkpoint_dir)
checkpoint_dir = cur_dir + c.checkpoint_dir + '/%s_%s_%s' % (lang1, lang2, c.attention)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
print 'INFO: checkpoints ready to go'


print 'INFO: training...'
lr = c.learning_rate
epochs = int(c.epochs)

best_valid_loss = float("inf")
train_losses = []
val_losses = []
try:
    for epoch in range(epochs):
        start = time.time()

        # training
        i = 0
        train_loss = 0.0
        for batch in tqdm(d.batch_iter(dataset='train'), total=d.num_batches('train')):
#        for batch in d.batch_iter(training=True):
            pred, loss, _ = model.train_on_batch(*batch, learning_rate=lr)
            i += 1.0
            train_loss += loss
        train_loss /= (i or 1)
        train_losses.append(train_loss)

        # validation
        i = 0.0
        val_loss = 0.0
        for batch in tqdm(d.batch_iter(dataset='val'), total=d.num_batches('val')):
#        for batch in d.batch_iter(training=False):
            pred, loss = model.run_on_batch(*batch, learning_rate=lr)
            val_loss += loss
            i += 1.0
        val_loss /= (i or 1)
        val_losses.append(val_loss)

        if epoch > 0 and val_loss < best_valid_loss:
            print 'INFO: new best validation loss!...'
            best_valid_loss = val_loss
            print 'INFO: generating plots...'
            filename = checkpoint_dir + '/%s-%s.png' % (c.attention, str(epoch))
            utils.lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                            [(train_losses, 'train'), (val_losses, 'val')])
            print 'INFO: plot saved to %s' % filename

            print 'INFO: saving checkpoint...'
            checkpoint_loc = checkpoint_dir + '/checkpoint-%s' % epoch
            model.save(checkpoint_loc)
            print 'INFO: checkpoint saved to %s' % (checkpoint_loc)

#        if epoch > 9 and epoch % 5 == 0 and lr > 0.0025:
#            lr = lr * 0.90

        print 'INFO: epoch,', epoch, 'train loss,', train_loss, 'val loss,', val_loss, 'time, ', (time.time() - start), 'lr: ' , lr

except KeyboardInterrupt:
    print 'INFO: stopped!'

finally:
    print 'INFO: generating plots...'
    filename = checkpoint_dir + '/final_losses.png'
    lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                    [(train_losses, 'train'), (val_losses, 'val')])
    print 'INFO: plot saved to %s' % filename


quit()
