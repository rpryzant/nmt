"""
Implementing an rnn with tensorflow


I'd like to make this model as close as possible to that in rnn.numpy.py:
   -2 layers: embedding, recursive
   -tanh on the hidden layer, softmax on predictions
   -initialize weights with random uniform on [-1/sqrt(input), 1/sqrt(input)]  
"""

import sys
import tensorflow as tf
import numpy as np
import sys

import random


VOCABULARY_SIZE = 8000 # from generate_data.py

class RNN(object):
    def __init__(self, input_dim, hidden_dim=100, backprop_clip=4, learning_rate=0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_clip = backprop_clip
        self.learning_rate = learning_rate

        # placeholders for a batch's worth of X and Y. each is the length of our bptt limit
        input_placeholder = tf.placeholder(tf.int32, shape=[None, self.bptt_clip], name="input") # or max seq len
        label_placeholder = tf.placeholder(tf.int32, shape=[None, self.bptt_clip], name="label")

        batch_size = tf.shape(input_placeholder)[0]

        input_lim = np.sqrt(1.0 / self.input_dim)
        hidden_lim = np.sqrt(1.0 / self.hidden_dim)

        # ==== layer 0: embedding
        U = tf.Variable(tf.random_uniform([self.input_dim, self.hidden_dim], -input_lim, input_lim))
        input = tf.nn.embedding_lookup(U, input_placeholder)
        input = tf.unstack(input, num=self.bptt_clip, axis=1)  # unroll examples into shortened list of tensors for bp truncation

        # ==== layer 1: rnn cell
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim, activation=tf.tanh)
        initial_state = tf.zeros([batch_size, rnn_cell.state_size])  # initialize hidden state with 0
        outputs, state = tf.nn.rnn(rnn_cell, input, initial_state=initial_state)

        # stack the outputs for each timestep and then flatten so that matrix multiplication will work nicely
        output2 = tf.reshape(tf.concat_v2(outputs, 1), [-1, self.hidden_dim])


        # ==== layer 2: fc on top (no bias because meh)
        V = tf.Variable(tf.random_uniform([self.hidden_dim, self.input_dim], -hidden_lim, hidden_lim))
        V_b = tf.Variable(tf.random_normal([self.input_dim]))
        # get pre-softmax predictions
        # the scan() is to multiply each batch by V individually
        logits = tf.matmul(output2, V) + V_b


        # training
        labels = tf.one_hot(label_placeholder, self.input_dim)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(loss)

        # session and variable initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self._input = input_placeholder
        self._label = label_placeholder
        self._rnn_cell = rnn_cell
        self._initial_state = initial_state
        self._state = state
        self._logits = logits
        self._loss = loss
        self._train_step = train_step
        self._session = sess

    # train on batch of complete sentances
    def train_on_batch(self, x, y):
        for window in range(len(x[0])):
            _, loss = self._session.run([self._train_step, self._loss],
                              feed_dict={
                                 self._input: [x[i][window] for i in range(len(x))],
                                 self._label: [y[i][window] for i in range(len(y))]
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

rnn = RNN(VOCABULARY_SIZE, backprop_clip=BACKPROP_CLIP)


X, Y = eval(open(sys.argv[1]).read())

X_train = X[:1000]
X_train = [[x[i:i+BACKPROP_CLIP] for i in range(len(x) - BACKPROP_CLIP + 1)] for x in X_train]

Y_train = Y[:1000]
Y_train = [[y[i:i+BACKPROP_CLIP] for i in range(len(y) - BACKPROP_CLIP + 1)] for y in Y_train]

print X_train[0]

for i in range(0, len(X_train) - BATCH_SIZE)[::BATCH_SIZE]:
    x_batch = X_train[i:i+BATCH_SIZE]
    y_batch = Y_train[i:i+BATCH_SIZE]

    loss = rnn.train_on_batch(x_batch, y_batch)

    if i % 100 == 0:
        print loss






