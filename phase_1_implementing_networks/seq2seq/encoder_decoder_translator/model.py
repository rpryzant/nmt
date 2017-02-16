"""
based on http://arxiv.org/abs/1412.7449 and https://arxiv.org/abs/1508.04025 (mostly the latter)

"""

import tensorflow as tf
import numpy as np





class Seq2Seq:
    def __init__(self, config, batch_size, testing=False, model_path=None):
        src_vocab_size    = config.src_vocab_size
        max_source_len    = config.max_source_len
        embedding_size    = config.embedding_size
        hidden_size       = config.hidden_size
        dropout_rate      = config.dropout_rate
        num_layers        = config.num_layers
        tgt_vocab_size    = config.tgt_vocab_size
        max_target_len    = config.max_target_len
        lr                = config.learning_rate

        self.testing = testing

        self.source     = tf.placeholder(tf.int32, shape=[None, max_source_len], name='source')
        self.target     = tf.placeholder(tf.int32, shape=[None, max_source_len], name='target')
        self.target_len = tf.placeholder(tf.int32, shape=[None], name='target_len')        # for masking loss

        # for counting interations
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


        # build encoder
        with tf.variable_scope("encoder"):
            # make all the graph nodes I'll need
            source_embedding = tf.get_variable("source_embedding",
                                               shape=[src_vocab_size, embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
            source_proj_W = tf.get_variable("s_proj_W", 
                                            shape=[embedding_size, hidden_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            source_proj_b = tf.get_variable("s_proj_b",
                                            shape=[hidden_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
            encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=dropout_rate)
            encoder = tf.nn.rnn_cell.MultiRNNCell([encoder_cell]*num_layers, state_is_tuple=True)

            # look up embedding
            source_x = tf.nn.embedding_lookup(source_embedding, self.source)
            source_x = tf.unstack(source_x, axis=1)                         # split into a list of embeddings, 1 per word

            # run encoder over source sentence
            s = encoder.zero_state(batch_size, tf.float32)
            for t in range(max_source_len):
                if t > 0: tf.get_variable_scope().reuse_variables()         # let tf reuse variables
                x = source_x[t]
                projection = tf.matmul(x, source_proj_W) + source_proj_b    # project embedding into rnn's space
                h, s = encoder(projection, s)
                
        # build decoder
        logits = []
        probs = []
        with tf.variable_scope("decoder"):
            target_embedding = tf.get_variable("target_embedding",
                                               shape=[tgt_vocab_size, embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
            target_proj_W = tf.get_variable("t_proj_W",
                                            shape=[embedding_size, hidden_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            target_proj_b = tf.get_variable("t_proj_b", shape=[hidden_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=dropout_rate)
            decoder = tf.nn.rnn_cell.MultiRNNCell([decoder_cell]*num_layers, state_is_tuple=True)

            # projection to output
            out_embed_W = tf.get_variable("o_embed_W",
                                          shape=[hidden_size, embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
            out_embed_b = tf.get_variable("o_embed_b",
                                          shape=[embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
            out_W = tf.get_variable("Wo", shape=[embedding_size, tgt_vocab_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            out_b = tf.get_variable("bo", shape=[tgt_vocab_size],
                                    initializer=tf.contrib.layers.xavier_initializer())

            # look up embedding
            target_x = tf.nn.embedding_lookup(target_embedding, self.target)
            target_x = tf.unstack(target_x, axis=1)

            # decode source by initializing with encoder's final hidden state
            for t in range(max_target_len):
                if t > 0: tf.get_variable_scope().reuse_variables()      # reuse variables after 1st iteration
                if not self.testing or t == 0: x = target_x[t]           # feed in provided targets while training

                projection = tf.matmul(x, target_proj_W) + target_proj_b # project embedding into rnn space
                h, s = decoder(projection, s)                            # s is last encoder state when t == 0
                
                out_embedding = tf.matmul(h, out_embed_W) + out_embed_b  # project output to target embedding space
                logit = tf.matmul(out_embedding, out_W) + out_b 
                logits.append(logit)
                prob = tf.nn.softmax(logit)
                probs.append(prob)
                
                if self.testing:
                    x = tf.cast(tf.argmax(prob, 1), tf.int32)
                    x = tf.nn.embedding_lookup(target_embedding, x)

        logits = logits[:-1]
        targets = tf.split(1, max_target_len, self.target)[1:]                # ignore <start> token
        target_mask = tf.sequence_mask(self.target_len - 1, max_target_len - 1, dtype=tf.float32)
        loss_weights = tf.unstack(target_mask, None, 1)                  # 0/1 weighting for variable len tgt seqs

        self.loss = tf.nn.seq2seq.sequence_loss(logits, targets, loss_weights)
        self.output_probs = tf.transpose(tf.pack(probs), [1, 0, 2])

        optimizer = tf.train.AdamOptimizer(lr)
        self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if model_path is not None:
            self.saver.restore(self.sess, model_path)

    
    def kill(self):
        """ kills this model
        """
        self.sess.close()


    def save(self, path):
        """ save the graph as a checkpoint.
            the provided path should include a filename prefix, as the
              written checkpoint will have an "iteration #" suffix
        """
        self.saver.save(self.sess, path, global_step=self.global_step)


    def predict_on_batch(self, x_batch):
        """ predict translation for a batch of inputs
        """
        if not self.testing:
            print "WARNING: model not in testing mode. previous outputs won't be fed back into the decoder. Reconsider"

        probs = self.sess.run(self.output_probs, feed_dict={self.source: x_batch})
        print probs
        return np.argmax(probs, axis=1)

    def train_on_batch(self, x_batch, y_batch, l_batch):
        """ train on a minibatch of data. x and y are assumed to be 
            padded to length max_seq_len, with l reflecting the original
            lengths of the target
        """
        _, loss = self.sess.run([self.train_step, self.loss],
                                feed_dict={
                                    self.source: x_batch,
                                    self.target: y_batch,
                                    self.target_len: l_batch
                                })
        return loss




# TODO - UNIT TESTS ON FAKE DATA
#c = config()
#test = Seq2Seq(c, 3)









