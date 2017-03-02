"""


=== DESCRIPTION
This is a basic seq2seq translation system without any 
bells or whistles. The model is in good shape, but this main
file is pretty horrid I admit. That's fine, though. If you run
it as-is, the model will overfit on a subset of the provided data. 


=== USAGE
python python main.py datasets/ [job name] [OPTIONAL : checkpoint file]
python main.py datasets/ test checkpoints_mine/checkpoint-11-12936
"""
import numpy as np
from data.data_iterator import Dataset
from models.seq2seq import Seq2SeqV3
import sys
import msc.utils
import time
from tqdm import tqdm
import os
from msc.utils import Config, lineplot, file_length, Progbar
import tensorflow as tf
from analysis.evaluation import *


data_loc = sys.argv[1]
job_id = sys.argv[2]
model_path = sys.argv[3] if len(sys.argv) > 3 else None


c = Config()
c.src_vocab_size = file_length(data_loc + c.x_vocab)    # hacky...make better
c.target_vocab_size = file_length(data_loc + c.y_vocab)


print 'INFO: building dataset...'
d = Dataset(c, data_loc)
i = 0
print 'INFO: dataset built. Train size: ', d.get_size('train'), 'val size: ', d.get_size('val')
#d.subset(13)    # take only X sentances
#print 'INFO: subset built. Train size: ', d.train_N, 'val size: ', d.val_N
print next(d.batch_iter(dataset='test'))[2][0]

print 'INFO: building checkpoint dir...'
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if not os.path.exists(cur_dir + c.checkpoint_dir):
    os.mkdir(cur_dir + c.checkpoint_dir)
checkpoint_dir = cur_dir + c.checkpoint_dir + '/%s' % job_id
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
print 'INFO: checkpoints ready to go'


with tf.Session() as sess:
    print 'INFO: building model...'
    model = Seq2SeqV3(c, d, sess, testing=True)
    if model_path is not None:
        model.load(filepath=model_path)
        print 'INFO: model loaded from %s' % model_path
    else:
        sess.run(tf.global_variables_initializer())
        print 'INFO: model built'


    print 'TESTING...'
    x_batch, x_lens, y_batch, _ = next(d.batch_iter(dataset='test'))
    ys = [d.reconstruct_target(y) for y in y_batch]
    yhats = [d.reconstruct_target(y) for y in model.predict_on_batch(x_batch, x_lens).tolist()]
    print multisentence_bleu(yhats, ys)
    print multisentence_ribes(yhats, ys)


tf.reset_default_graph()


with tf.Session() as sess:
    print 'INFO: building a SECOND model...'
    model = Seq2SeqV3(c, d, sess, testing=True)
    if model_path is not None:
        model.load(filepath=model_path)
        print 'INFO: model loaded from %s' % model_path
    else:
        sess.run(tf.global_variables_initializer())
        print 'INFO: model built'



    print 'TESTING...'
    x_batch, x_lens, _, _ = next(d.batch_iter(dataset='test'))
    print y_batch[0]
    print model.predict_on_batch(x_batch, x_lens)[0][0]

tf.reset_default_graph()

quit()



with tf.Session() as sess:
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
            prog = Progbar(target=d.num_batches('train'))
            train_loss = 0.0
    #        for batch in tqdm(d.batch_iter(dataset='train'), total=d.num_batches('train')):
            for batch in d.batch_iter(dataset='train'):
                pred, loss, _ = model.train_on_batch(*batch, learning_rate=lr)
                i += 1.0
                prog.update(i, [('train loss', loss)])
                train_loss += loss
            train_loss /= (i or 1)
            train_losses.append(train_loss)

            # validation
            i = 0.0
            prog = Progbar(target=d.num_batches('val'))        
            val_loss = 0.0
    #        for batch in tqdm(d.batch_iter(dataset='val'), total=d.num_batches('val')):
            for batch in d.batch_iter(dataset='val'):
                pred, loss = model.run_on_batch(*batch, learning_rate=lr)
                val_loss += loss
                i += 1.0
                prog.update(i, [('val loss', loss)])
            val_loss /= (i or 1)
            val_losses.append(val_loss)

            if epoch > 0 and val_loss < best_valid_loss:
                print 'INFO: new best validation loss!...'
                best_valid_loss = val_loss
                print 'INFO: generating plots...'
                filename = checkpoint_dir + '/%s-%s.png' % (c.attention, str(epoch))
                lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
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
