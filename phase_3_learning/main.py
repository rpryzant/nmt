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
from msc import utils
import time
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



print 'INFO: building run dir and logger...'
def make_dir(p):
    if not os.path.exists(p):
        os.mkdir(p)
cur_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
run_dir = os.path.join(cur_dir, job_id)
checkpoint_dir = os.path.join(run_dir, c.checkpoint_dir)
fig_dir = os.path.join(run_dir, c.fig_dir)
result_dir = os.path.join(run_dir, c.result_dir)
make_dir(run_dir)
make_dir(checkpoint_dir)
make_dir(fig_dir)
make_dir(result_dir)
LOGGER = utils.Logger(os.path.join(run_dir, 'log'))

print 'INFO: run dir built at [%s]' % run_dir
print "i'll be shutting up now. Bye bye."
print 'logs at %s' % os.path.join(result_dir, 'log')




#try:
with tf.Session() as sess:
    LOGGER.log('INFO: building train model...')
    model = Seq2SeqV3(c, d, sess, testing=False)
    if model_path is not None:
        model.load(filepath=model_path)
        LOGGER.log('INFO: model loaded from %s' % model_path)
    else:
        sess.run(tf.global_variables_initializer())
        LOGGER.log('INFO: model built')

    LOGGER.log('INFO: training...')
    lr = c.learning_rate
    epochs = int(c.epochs)

    best_valid_loss = float("inf")
    train_losses = []
    val_losses = []



    for epoch in range(1):
        start = time.time()

        # training
        i = 0
        prog = Progbar(target=d.num_batches('train'))
        train_loss = 0.0
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
        for batch in d.batch_iter(dataset='val'):
            pred, loss = model.run_on_batch(*batch, learning_rate=lr)
            val_loss += loss
            i += 1.0
            prog.update(i, [('val loss', loss)])
        val_loss /= (i or 1)
        val_losses.append(val_loss)

        if epoch > 0 and val_loss < best_valid_loss:
            LOGGER.log('INFO: new best validation loss!...')
            best_valid_loss = val_loss
            LOGGER.log('INFO: generating plots...')
            filename = fig_dir + '/%s-%s.png' % (c.attention, str(epoch))
            lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                            [(train_losses, 'train'), (val_losses, 'val')])
            LOGGER.log('INFO: plot saved to %s' % filename)

            LOGGER.log('INFO: saving checkpoint...')
            checkpoint_loc = checkpoint_dir + '/checkpoint-%s' % epoch
            model.save(checkpoint_loc)
            LOGGER.log('INFO: checkpoint saved to %s' % (checkpoint_loc))


        seconds = (time.time() - start)
        seconds_per_batch = seconds / (d.num_batches('train') + d.num_batches('val'))
        LOGGER.log('INFO: epoch,', epoch, 'train loss,', train_loss, 'val loss,', \
                    val_loss, 'time, ', seconds, 'per batch ', seconds_per_batch, \
                    'lr: ' , lr)
        print 'HERE'
        model.save(os.path.join(checkpoint_dir,'checkpoint'))
        print 'SAVED'

quit()
try:
    pass
except KeyboardInterrupt:
    print 'STOPPED!'
    tf.reset_default_graph()
finally:
    tf.reset_default_graph()

    with tf.Session() as sess:
        YHAT_WRITER = utils.Logger(os.path.join(result_dir, 'yhats'))
        Y_WRITER = utils.Logger(os.path.join(result_dir, 'ys'))
        LOGGER.log('INFO: building test model...')
        model = Seq2SeqV3(c, d, sess, testing=True)
        model.load(dir=checkpoint_dir)
        LOGGER.log('INFO: model loaded from %s' % checkpoint_dir)            

        LOGGER.log('TESTING...')
        yhats = []
        ys = []
        i = 0
        prog = Progbar(target=d.num_batches('test'))
        for x_batch, x_lens, y_batch, _ in d.batch_iter(dataset='test'):
            ys += [d.reconstruct_target(y) for y in y_batch]
            yhat = model.predict_on_batch(x_batch, x_lens).tolist()
            yhats += [d.reconstruct_target(y) for y in yhat]
            prog.update(i, [])
            i += 1

        YHAT_WRITER.log('\n'.join(' '.join(x.decode('utf-8') for x in s) for s in yhats),
                        show_time=False)
        Y_WRITER.log('\n'.join(' '.join(x.decode('utf-8') for x in s) for s in yhats),
                     show_time=False)

        LOGGER.log('RESULT: final BLEU: ', multisentence_bleu(ys, yhats)) 
        LOGGER.log('RESULT: final RIBES: ', multisentence_ribes(ys, yhats))



    LOGGER.log('INFO: generating plots...')
    filename = fig_dir + '/final_losses.png'
    lineplot(filename, 'Train/Val Losses', 'epoch', 'Loss', 
                    [(train_losses, 'train'), (val_losses, 'val')])
    LOGGER.log('INFO: plot saved to %s' % filename)


quit()
