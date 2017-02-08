"""
based on http://arxiv.org/abs/1412.7449 and https://arxiv.org/abs/1508.04025 (mostly the latter)


FINISHED READING DILLIONS CODE. LOOKS GOOD. UNDERSTAND PRETTY MUCH EVERYTHING EXCEPT
FOR A FEW TF MATRIX MANIPULATIONS. EASY ENOUGH TO FIGURE OUT WHEN YOU'RE PLAYING
WIT THE PLAYDOUGH. GONNA START THIS!!

*********ONE THING YOU SHOULD DO, REID, IS USE TF.UNPACK INSTEAD OF HIS SPLITTING AND SMUSHING
            TO LOOP THROUGH ALL THE EMBEDDING VECTORS FOR A SENTANCE




=== ATTENTION RESOURCES:

http://distill.pub/2016/augmented-rnns/
https://github.com/lmthang/thesis/blob/master/thesis.pdf
  pg 54
https://arxiv.org/abs/1508.04025
http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
https://blog.heuritech.com/2016/01/20/attention-mechanism/
https://indico.io/blog/sequence-modeling-neural-networks-part2-attention-models/

"""

import tensorflow as tf
import numpy as np



class config:
    source_n = 80000
    max_source_len = 25
    embedding_size = 64
    hidden_size = 128
    dropout_rate = 0.5
    num_layers = 3
    target_n = 80000
    max_target_len = 25
    learning_rate = 0.0003



class Seq2Seq:
    def __init__(self, config, batch_size, testing=False):
        source_n          = config.source_n
        max_source_len    = config.max_source_len
        embedding_size    = config.embedding_size
        hidden_size       = config.hidden_size
        dropout_rate      = config.dropout_rate
        num_layers        = config.num_layers
        target_n          = config.target_n
        max_target_len    = config.max_target_len
        lr                = config.learning_rate

        self.testing = testing

        source     = tf.placeholder(tf.int32, [batch_size, max_source_len], name='source')
        target     = tf.placeholder(tf.int32, [batch_size, max_source_len], name='target')
        target_len = tf.placeholder(tf.int32, [batch_size], name='target_len')        # for masking loss


        # build encoder
        with tf.variable_scope("encoder"):
            # make all the graph nodes I'll need
            source_embedding = tf.get_variable("source_embedding",
                                               shape=[source_n, embedding_size],
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
            source_x = tf.nn.embedding_lookup(source_embedding, source)
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
                                               shape=[target_n, embedding_size],
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
            out_W = tf.get_variable("Wo", shape=[embedding_size, target_n],
                                    initializer=tf.contrib.layers.xavier_initializer())
            out_b = tf.get_variable("bo", shape=[target_n],
                                    initializer=tf.contrib.layers.xavier_initializer())

            # look up embedding
            target_x = tf.nn.embedding_lookup(target_embedding, target)
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
        targets = tf.split(1, max_target_len, target)[1:]                # ignore <start> token
        target_mask = tf.sequence_mask(target_len - 1, max_target_len - 1, dtype=tf.float32)
        loss_weights = tf.unstack(target_mask, None, 1)                  # 0/1 weighting for variable len tgt seqs

        loss = tf.nn.seq2seq.sequence_loss(logits, targets, loss_weights)
        output_probs = tf.transpose(tf.pack(probs), [1, 0, 2])

        optimizer = tf.train.AdamOptimizer(lr)
        train_step = optimizer.minimize(loss)


c = config()

test = Seq2Seq(c, 3)
