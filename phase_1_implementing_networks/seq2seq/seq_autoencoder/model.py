"""
Simplest possible seq2seq model I could think of. Good way to get the toes wet. 

This is a seq2seq model that works like an autoencoder: given a sequence of vectors X,
  it first encodes into a single vector representation, then decodes into the exact
  same sequence X.

https://www.tensorflow.org/tutorials/seq2seq/

TODO: 
  - REFACTOR INTO CLASS
  - BETTER DOCUMENTATION
  - TRY FEEDING PREVIOUS - REGENERATE SEQUENCE FROM ENCODER HIDDEN STATE
"""


import tensorflow as tf
import numpy as np



class SeqAutoencoder(object):

    def __init__(self, seq_length, vocab_size, embedding_dim, memory_dim):
        # inputs for encoder
        encoder_input = [tf.placeholder(tf.int32, shape=[None]) for t in range(seq_length)]

        # prepend "start" token and drop the encoder's final input. otherwise you're trying to decode the same thing
        decoder_input = ( [tf.zeros_like(encoder_input[0], dtype=tf.int32)] + encoder_input[:-1] )

        # autoencoder, so trying to recreate inputs
        decoder_labels = [tf.placeholder(tf.int32, shape=(None,)) for t in range(seq_length)]

        # dumb initialization but meh
        weights = [tf.ones_like(label_tensor, dtype=tf.float32) for label_tensor in decoder_labels]

        # make lstm cell
        cell = tf.nn.rnn_cell.LSTMCell(memory_dim)

        # set up embedding rnn (takes care of word vectors for us) 
        # make seq2seq model 
        # set feed_previous=False. don't want to feed previous outputs back into the model
        decoder_outputs, decoder_states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            encoder_input, decoder_input, cell, vocab_size, vocab_size, embedding_dim)

        # cross entropy loss
        loss = tf.nn.seq2seq.sequence_loss(decoder_outputs, decoder_labels, weights, vocab_size)

        # training
        learning_rate = 0.05
        momentum = 0.9
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_step = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        self.__seq_len = seq_length          # no dynamic sequence length because meh
        self.__input = encoder_input
        self.__labels = decoder_labels
        self.__train_step = train_step
        self.__loss = loss
        self.__sess = sess

    def train_on_batch(self, x_batch, y_batch):
        # slot in the expected inputs and outputs
        feed_dict = {self.__input[t]: x_batch[t] for t in range(self.__seq_len)}
        feed_dict.update({self.__labels[t]: y_batch[t] for t in range(self.__seq_len)})

        # run teh training step!
        _, loss = self.__sess.run([self.__train_step, self.__loss], feed_dict)

        return loss
                         

def feed_batch(model, batch_size, vocab_size):
    """ feeds a batch of random data into a given model
    """
    X = [np.random.choice(vocab_size, size=(seq_length,)) for _ in range(batch_size)]
    Y = X[:]

    # turn each example into a column vector
    X = np.array(X).T
    Y = np.array(Y).T

    loss = model.train_on_batch(X, Y)

    return loss




seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100



model = SeqAutoencoder(seq_length, vocab_size, embedding_dim, memory_dim)

for t in range(500):
    X = [np.random.choice(vocab_size, size=(seq_length,)) for _ in range(batch_size)]
    Y = X[:]

    # turn each example into a column vector
    X = np.array(X).T
    Y = np.array(Y).T

    loss = model.train_on_batch(X, Y)

    print loss
