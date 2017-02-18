"""




python main.py data/processed/ en vi [checkpoint prefix for saving or loading] [restore if you want to restore]

python main.py data/processed/ en vi tmp/checkpoint.ckpt
python main.py data/processed/ en vi tmp/checkpoint.ckpt-103 load

"""

from dataset import Dataset
from model import Seq2Seq, Seq2SeqV2, Seq2SeqV3
import sys



class config:
    src_vocab_size = 5000 + 1 # +1 for unk
    max_source_len = 50
    embedding_size = 64
    hidden_size = 128
    dropout_rate = 0.5
    num_layers = 3
    target_vocab_size = 5000 + 1 # +1 for unk
    max_target_len = 50
    learning_rate = 0.0001


data_loc = sys.argv[1]
lang1 = sys.argv[2]
lang2 = sys.argv[3]
model_path = sys.argv[4]
restore = sys.argv[5] if len(sys.argv) == 6 else None


batch_size = 5
print 'building dataset...'
d = Dataset(data_loc, lang1, lang2)
'dataset done'
d.subset(50)    # take only 2k sentances

c = config()



# if restore is not None:
#     print 'restoring model...'
#     model = Seq2Seq(c, batch_size, testing=True, model_path=model_path)
#     print 'model restored.'
# else:
#     print 'no model found. building model...'
#     model = Seq2Seq(c, batch_size)
#     print 'model built.'
print 'building model...'
model = Seq2SeqV3(c, batch_size)
print 'model built.'

print 'training...'
for epoch in range(7):
    epoch_loss = 0.0
    i = 0
    while d.has_next_batch(batch_size):
        loss = model.train_on_batch(*d.next_batch(batch_size))
        epoch_loss += loss
        i += 1
    print 'epoch', epoch, 'loss', (epoch_loss / (i * batch_size))
    d.reset()


d.reset()

batch = d.next_batch(batch_size)    # extract x's
pred = model.predict_on_batch(*batch)

print [np.argmax(p) for p in pred[0]]

for i in range(len(batch)):
    print d.reconstruct(batch[i][0], lang1)
    print
    print batch[i][0]
    print d.reconstruct(batch[i][0], lang2)
    print
    print pred[i]
    print d.reconstruct(pred[i], lang2)

    print '==================================='

quit()





try:
    print 'training...'
    for epoch in range(20):
        epoch_loss = 0.0
        i = 0
        while d.has_next_batch(batch_size):
            loss = model.train_on_batch(*d.next_batch(batch_size))
            epoch_loss += loss
            i += 1
        print 'epoch={}\t mean batch loss={:.4f}'.format(epoch, epoch_loss / (i * batch_size))
        d.reset()



except KeyboardInterrupt:
    print 'saving model to', model_path
    save_path = model.save('./%s' % model_path)
    print 'saved at ', save_path

finally:
        print 'testing...'
        d.reset()
        
        test_batch = d.next_batch(batch_size)  
        pred_indices = model.predict_on_batch(*test_batch)
        
        for (y, y_hat) in zip(test_batch[0], pred_indices):
            print d.reconstruct(y, lang2)
            print d.reconstruct(y_hat, lang2)




