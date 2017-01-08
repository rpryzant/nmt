"""
My tf lstm had some weird behavior at first (losses bouncing around regaurdless of learning rate)
 so I'm going to spend some time stripping my implementation down until I get something that works.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
from data_generator import DataGenerator

class LSTM(object):
    
    def __init__(self, num_classes, max_seq_len, learning_rate, hidden_units):
        # placeholders for data
        input_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        target_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        length_placeholder = tf.placeholder(tf.int32, shape=[None])

        # retrieve and unpack input embeddings
        U = tf.get_variable(
            name="U",
            initializer=tf.random_normal_initializer(),
            shape=[num_classes, hidden_units])
        input = tf.nn.embedding_lookup(U, input_placeholder)


        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units)
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=input,
            sequence_length=length_placeholder,
#            initial_state=cell.zero_state(batch_size, tf.float32),
            dtype=tf.float32)

        # output layer
        V = tf.get_variable(
            name='V',
            initializer=tf.random_normal_initializer(),
            shape=[hidden_units, num_classes])

        # smush together all the batches so that we can calculate loss in one step
        outputs = tf.reshape(outputs, [-1, hidden_units])    
        logits = tf.matmul(outputs, V)

        # loss
        target_flat = tf.reshape(target_placeholder, [-1])  # flatten out batches for same reason as above
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target_flat)

        # mask out padded targets from loss
        mask = tf.sign(tf.to_float(target_flat)) 
        losses = mask * losses

        # recover batches and compute mean loss per batch 
        losses = tf.reshape(losses, tf.shape(target_placeholder))
        loss_per_batch = tf.reduce_sum(losses, reduction_indices=1) / tf.to_float(length_placeholder)
        mean_batch_loss = tf.reduce_mean(loss_per_batch)

        # training
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003)
        train_step = optimizer.minimize(mean_batch_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self._input = input_placeholder
        self._target = target_placeholder
        self._length = length_placeholder
        self._sess = sess
        self._train_step = train_step
        self._loss = mean_batch_loss

    def train_on_batch(self, x_batch, y_batch, l_batch):
        _, loss = self._sess.run([self._train_step, self._loss],
                           feed_dict={
                               self._input: x_batch,
                               self._target: y_batch,
                               self._length: l_batch
                           })
        return loss






MAX_SEQ_LEN = 200
BATCH_SIZE = 2
VOCAB_SIZE = 8001 # 1 extra for padding

dg = DataGenerator(sys.argv[1], 4) # batch size of 4

training_data = [dg.next_batch() for _ in range(100)]  # pull out 100 batches


lstm = LSTM(VOCAB_SIZE, MAX_SEQ_LEN, 0.0003, 128)

for epoch in range(1000):
    epoch_loss = 0
    for x_batch, y_batch, l_batch in tqdm(training_data):
        epoch_loss += lstm.train_on_batch(x_batch, y_batch, l_batch)

    print epoch_loss
                           

                           
