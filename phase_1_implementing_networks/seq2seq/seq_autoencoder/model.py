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






seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100


encoder_input = [tf.placeholder(tf.int32, shape=[None]) for t in range(seq_length)]

# prepend "start" token and drop the encoder's final input
decoder_input = ( [tf.zeros_like(encoder_input[0], dtype=tf.int32)] + encoder_input[:-1] )

decoder_labels = [tf.placeholder(tf.int32, shape=(None,)) for t in range(seq_length)]

# better initialization...
weights = [tf.ones_like(label_tensor, dtype=tf.float32) for label_tensor in decoder_labels]

# initial values for recurrence backprop
memory = tf.zeros((batch_size, memory_dim))

cell = tf.nn.rnn_cell.LSTMCell(memory_dim)
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


def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,)) for _ in range(batch_size)]
    Y = X[:]

    # turn each example into a column vector
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {encoder_input[t]: X[t] for t in range(seq_length)}
    feed_dict.update({decoder_labels[t]: Y[t] for t in range(seq_length)})

    _, step_loss = sess.run([train_step, loss], feed_dict)

    return step_loss


for t in range(500):
    loss_t = train_batch(batch_size)
    print loss_t
