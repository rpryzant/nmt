"""
Implementing an lstm with tensorflow


I'd like to make this model as close as possible to that in lstm.numpy.py:
   -2 layers: embedding, recursive
   -tanh on the hidden layer, softmax on predictions
   -initialize weights with random uniform on [-1/sqrt(input), 1/sqrt(input)]  

***NOTE***
This program is pretty much identical to ../rnn/rnn_tf.py with two big exceptions:
 1) BasicRNNCell has been swaped out with BasicLSTMCell. Easy peasy!
 2) I'm using tf.nn.dynamic_rnn in lieu of tf.nn.rnn to unroll the graph. tf.nn.dynamic_rnn
     uses a while loop, and keeps unrolling as long as you have more timesteps in your example. 
     tf.nn.rnn, on the other hand, has to be unrolled by a set number of steps (thereby 
     clipping backprop). The reason I'm doing this here is
     (1) the LSTM can propagate gradients further backwards, and (2) because of #1, the lstm can capture
     longer-range dependancies (i.e. how the 1st word in a sentance can effect the 20th). So 
     lstm's can actually handle entire sentances. Because this, I'm not breaking each example
     sentance into a sliding window of words either. just feeding the whole thing in.

"""

import sys
import tensorflow as tf
import numpy as np
import sys
from tqdm import tqdm 
import random


MAX_LENGTH = 250
VOCABULARY_SIZE = 8001 # 8000 from generate_data.py, and 1 for padding

class LSTM(object):
    def __init__(self, input_dim, hidden_dim=100, backprop_clip=4, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_clip = backprop_clip
        self.learning_rate = learning_rate

        # placeholders for a batch's worth of X and Y. each is the length of our bptt limit
        input_placeholder = tf.placeholder(tf.int32, shape=[None, MAX_LENGTH], name="input") # or max seq len
        label_placeholder = tf.placeholder(tf.int32, shape=[None, MAX_LENGTH], name="label")
        input_lengths = tf.placeholder(tf.int32, shape=[None], name="lengths")

        batch_size = tf.shape(input_placeholder)[0]

        input_lim = np.sqrt(1.0 / self.input_dim)
        hidden_lim = np.sqrt(1.0 / self.hidden_dim)

        # ==== layer 0: embedding
        U = tf.Variable(tf.random_uniform([self.input_dim, self.hidden_dim], -input_lim, input_lim))
        input = tf.nn.embedding_lookup(U, input_placeholder)
        input = tf.unpack(input, axis=1)  # unroll examples into list of tensors 

        # ==== layer 1: rnn cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, activation=tf.tanh)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)  # initialize hidden state with 0

        outputs, state = tf.nn.rnn(
            lstm_cell, 
            input, 
            initial_state=initial_state, 
            sequence_length=input_lengths)

        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2]) # pack outputs into a tensor of shape [batch size, max timesteps, hidden dim]

        # ==== layer 2: fc on top (no bias because meh)
        V = tf.Variable(tf.random_uniform([self.hidden_dim, self.input_dim], -hidden_lim, hidden_lim))
        V_b = tf.Variable(tf.random_normal([self.input_dim]))
        # get pre-softmax predictions
        rnn_outputs_flat = tf.reshape(outputs, [-1, self.hidden_dim])
        logits = tf.batch_matmul(rnn_outputs_flat, V) + V_b
        
        labels_flat = tf.reshape(label_placeholder, [-1])

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_flat)
        # mask losses
        mask = tf.sign(tf.to_float(labels_flat))  # this is why we bumped up all the classes
        masked_losses = mask * losses
        # bring back to [batches, max timesteps]
        masked_losses = tf.reshape(masked_losses, tf.shape(input_placeholder))
        # calc mean loss
        mean_loss = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(input_lengths)
        mean_loss = tf.reduce_mean(mean_loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(mean_loss)

        # session and variable initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self._input = input_placeholder
        self._label = label_placeholder
        self._lstm_cell = lstm_cell
        self._initial_state = initial_state
        self._state = state
        self._logits = logits
        self._loss = mean_loss
        self._train_step = train_step
        self._session = sess
        self._lengths = input_lengths

    # train on batch of complete sentances
    def train_on_batch(self, x, y):
        for batch_i in range(len(x)):
            # increment everything by one so that 0 gets its own reserved padding id
            for j in range(len(x[batch_i])):
                x[batch_i][j] += 1
                y[batch_i][j] += 1
            # 0-pad
            l = len(x[batch_i])
            while l < MAX_LENGTH:
                x[batch_i].append(0)
                y[batch_i].append(0)
                l += 1
        if len(x[0]) == 0:
            return

        lengths = lambda l: [len(y) for y in l]

        _, loss = self._session.run([self._train_step, self._loss],
                          feed_dict={
                             self._input: x,
                             self._label: y,
                             self._lengths: lengths(x) # true sentance lengths
                          })
        return loss

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._state

    @property
    def loss(self):
        return self._loss

    @property
    def train_step(self):
        return self._train_step

    @property
    def session(self):
        return self._session

    
        
        

BATCH_SIZE = 1
BACKPROP_CLIP = 5

lstm = LSTM(VOCABULARY_SIZE, backprop_clip=BACKPROP_CLIP)


X, Y = eval(open(sys.argv[1]).read())

X_train = X[:1000]
# no need to break up into sliding window since we're doing variable-length inputs
#X_train = [[x[i:i+BACKPROP_CLIP] for i in range(len(x) - BACKPROP_CLIP + 1)] for x in X_train]

Y_train = Y[:1000]
#Y_train = [[y[i:i+BACKPROP_CLIP] for i in range(len(y) - BACKPROP_CLIP + 1)] for y in Y_train]


for epoch in range(1000):
    epoch_loss = 0.0
    for i in tqdm(range(0, len(X_train) - BATCH_SIZE)[::BATCH_SIZE]):
        x_batch = X_train[i:i+BATCH_SIZE]
        y_batch = Y_train[i:i+BATCH_SIZE]
        if any(x == [] for x in x_batch):  # sometimes the parser makes a boo boo
            continue
        loss = lstm.train_on_batch(x_batch, y_batch)

        epoch_loss += loss
    print '======== epoch %d. cumulative loss: %d' % (epoch, epoch_loss)






