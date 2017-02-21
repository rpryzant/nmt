"""
based on https://arxiv.org/abs/1406.1078

TODO: embed start symbol at test time
"""

import tensorflow as tf
import numpy as np




class Seq2SeqV3(object):
    def __init__(self, config, batch_size, testing=False, model_path=None):
        # configurations
        self.src_vocab_size       = config.src_vocab_size
        self.max_source_len       = config.max_source_len
        self.embedding_size       = config.embedding_size
        self.hidden_size          = config.hidden_size
        self.num_layers           = config.num_layers
        self.target_vocab_size    = config.target_vocab_size

        # args
        self.batch_size           = batch_size
        self.testing              = testing

        # placeholders
        self.learning_rate = tf.placeholder(tf.int32, shape=(), name="lr")
        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="source")
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name="source_len")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="target")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # get word vectors for the source and target
        source_embedded, target_embedded = self.get_embeddings(self.source, self.target)

        # run everything through the encoder and decoder
        self.decoder_output = self.encode_decode(source=source_embedded,
                                                 source_len=self.source_len,
                                                 target=target_embedded,
                                                 target_len=self.target_len)

        # compute average per-word loss across all batches (log perplexity)
        self.loss = self.cross_entropy_sequence_loss(logits=self.decoder_output,
                                                targets=self.target,
                                                seq_len=self.target_len)

        # compute and apply gradients
        self.train_step = self.backward_pass(self.loss)

        # tf boilerplate
        self.check = tf.add_check_numerics_ops()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
#        self.writer =  tf.summary.FileWriter('./graphs', self.sess.graph)
#       self.writer.close()


    def get_embeddings(self, source, target):
        """ source: [batch size, max length]    = source-side one-hot vectors
            target: [batch size, max length]    = target-side one-hot vectors

            returns word embeddings for each item in the source/target sequence
        """
        source_embedding = tf.get_variable('source_embeddings',
                                           shape=[self.src_vocab_size, self.embedding_size])
        source_embedded = tf.nn.embedding_lookup(source_embedding, source)

        target_embedding = tf.get_variable('target_embeddings',
                                           shape=[self.target_vocab_size, self.embedding_size])
        target_embedded = tf.nn.embedding_lookup(target_embedding, target)

        return source_embedded, target_embedded


    def train_on_batch(self, x_batch, x_lens, y_batch, y_lens, learning_rate=1.0):
        """ train on a minibatch of data. x and y are assumed to be 
            padded to length max_seq_len, with [x/y]_lens reflecting
            the original lengths of each sequence
        """
        _, logits, loss = self.sess.run([self.train_step, self.decoder_output, self.loss],
                                feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.target: y_batch,
                                    self.target_len: y_lens,
                                    self.dropout: 0.5,
                                    self.learning_rate: learning_rate
                                })

        return np.argmax(logits, axis=2), loss


    def predict_on_batch(self, x_batch, x_lens, y_batch, y_lens):
        """ predict translation for a batch of inputs

            TODO - only take x_batch, and feed in the start symbol instead of
                    the first word from y (which is start symbol)
        """
        if not self.testing:
            print "WARNING: model not in testing mode. previous outputs won't be fed back into the decoder. Reconsider"

        logits = self.sess.run(self.decoder_output, feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.target: y_batch,
                                    self.target_len: y_lens,
                                    self.dropout: 0.5
                                })

        return np.argmax(logits, axis=2), logits


    def backward_pass(self, loss):
        """ use the given loss to construct a training step 
            NOTE: Using SGD instead of adagrad or adam because those don't seem to work
        """
        train_step = tf.contrib.layers.optimize_loss(self.loss, None,
                self.learning_rate, "SGD", clip_gradients=5.0)
        return train_step


    def cross_entropy_sequence_loss(self, logits, targets, seq_len):
        """ logits:  [batch size, sequence len, vocab size]
            targets: [batch size, sequence len]
            lengths: [batch size]        = length of each target sequence before padding

            computes and returns per-timestep cross-entropy, then averages across sequences and batches
        """
        targets = targets[:, 1:]              # shift targest forward by 1: ignore start symbol
        logits = logits[:, :-1, :]            # take off last group of logits so dimensions match up
        seq_len = seq_len - 1                 # update sequence lengths to reflect this
        
        # cross entropy
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=targets)

        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(seq_len, targets.get_shape()[1], dtype=tf.float32)
        losses = losses * loss_mask

        # get mean log perplexity across all batches
        loss = tf.reduce_sum(losses) / tf.to_float(tf.reduce_sum(seq_len))
        return loss


    def encode_decode(self, source, source_len, target, target_len):
        """ source: [batch size, sequence len]
            source_len: [batch size]          = pre-padding lengths of source sequences
            target: [batch size, sequence len]
            target_len: [batch size]          = pre-padding lengths of targets

            runs the source through an encoder, then runs decoder on final hidden state
        """
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_rnn_cell()        
            encoder_output = self.run_encoder(source, source_len, encoder_cell)

        with tf.variable_scope('decoder'):
            decoder_cell = self.build_rnn_cell()
            decoder_output = self.run_decoder(target, 
                                              target_len, 
                                              decoder_cell, 
                                              encoder_output['final_state'])
        return decoder_output


    def run_decoder(self, target, target_len, decoder, initial_state):
        """ target: [batch size, sequence len]
            target_len: [batch_size]  = pre-padded target lengths
            decoder: RNNCell 
            initial_state: tuple([batch size, hidden state size]  * batch size  )

            runs a decoder on target and returns its predictions at each timestep
        """
        # projection to rnn space
        target_proj_W = tf.get_variable("t_proj_W",
                                        shape=[self.embedding_size, self.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        target_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        # projection to output embeddings
        out_embed_W = tf.get_variable("o_embed_W",
                                      shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        out_embed_b = tf.get_variable("o_embed_b",
                                      shape=[self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        # projection to logits
        out_W = tf.get_variable("Wo", shape=[self.embedding_size, self.target_vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())
        out_b = tf.get_variable("bo", shape=[self.target_vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())


        # decode source by initializing with encoder's final hidden state
        target_x = tf.unstack(target, axis=1)                        # break the target into a list of word vectors
        s = initial_state                                            # intial state = encoder's final state
        logits = []
        for t in range(target.get_shape()[1]):
            if t > 0: tf.get_variable_scope().reuse_variables()      # reuse variables after 1st iteration
            if self.testing and t == 0:
                # TODO EMBED SEQUENCE START TOKEN AND STICK IT INTO S
                raise NotImplementedError
            elif not self.testing:
                x = target_x[t]                                      # while training, feed in correct input

            projection = tf.matmul(x, target_proj_W) + target_proj_b # project embedding into rnn space
            h, s = decoder(projection, s)                            # s is last encoder state when t == 0
            
            out_embedding = tf.matmul(h, out_embed_W) + out_embed_b  # project output to target embedding space
            logit = tf.matmul(out_embedding, out_W) + out_b 
            logits.append(logit)

            if self.testing:
                x = tf.cast(tf.argmax(prob, 1), tf.int32)            # if testing, use cur pred as next input
                x = tf.nn.embedding_lookup(target_embedding, x)

        logits = tf.stack(logits)
        logits = tf.transpose(logits, [1, 0, 2])                     # get into [batch size, sequence len, vocab size]
        return logits


    def run_encoder(self, source, source_len, cell):
        """ source: [batch size, seq len]
            source_len: [batch size]   = pre-padding lengths
            cell: RNNCell

            runs the cell inputs for source-len timesteps
        """
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=source,
            sequence_length=source_len,
            dtype=tf.float32)
        return {'outputs': outputs, 'final_state': state}


    def build_rnn_cell(self):
        """ builds a stacked RNNCell according to specs defined in the model's config
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                             input_keep_prob=self.dropout,
                                             output_keep_prob=self.dropout)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
        return stacked_cell







