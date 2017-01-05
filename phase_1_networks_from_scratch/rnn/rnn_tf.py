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

VOCABULARY_SIZE = 8000 # from generate_data.py


class RNN(object):
    def __init__(self, input_dim, hidden_dim=100, bptt_clip=4, learning_rate=0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_clip = bptt_clip
        self.learning_rate = learning_rate

        # placeholders for a batch's worth of X and Y. each is the length of our bptt limit
        input = tf.placeholder(tf.int32, shape=[None, self.bptt_clip], name="input") # or max seq len
        label = tf.placeholder(tf.int32, shape=[None, self.input_dim], name="label")
        batch_size = tf.shape(input)[0]
        input_lim = np.sqrt(1.0 / self.input_dim)
        hidden_lim = np.sqrt(1.0 / self.hidden_dim)

        # ==== layer 0: embedding
        U = tf.Variable(tf.random_uniform([self.input_dim, self.hidden_dim], -input_lim, input_lim))
        input = tf.nn.embedding_lookup(U, input)
        input = tf.unstack(input, num=self.bptt_clip, axis=1)  # unroll examples into shortened list of tensors for bp truncation

        # ==== layer 1: rnn cell
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim, activation=tf.tanh)
        self.initial_state = tf.zeros([batch_size, rnn_cell.state_size])  # initialize hidden state with 0
        outputs, state = tf.nn.rnn(rnn_cell, input, initial_state=self.initial_state)

        # ==== layer 2: fc on top (no bias because meh)
        V = tf.Variable(tf.random_uniform([self.hidden_dim, self.input_dim], -hidden_lim, hidden_lim))
        V_b = tf.Variable(tf.random_normal([self.input_dim]))
        logits = tf.add(tf.matmul(state, V), V_b)  # predictions before softmax

        # training
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, label))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(loss)

#        x = tf.one_hot(inputs, VOCABULARY_SIZE)     # transform x into 1-hot vector
#        x = tf.unpack(x, axis=1)                    # break x into a list of sequential inputs
        
        
rnn = RNN(VOCABULARY_SIZE)
