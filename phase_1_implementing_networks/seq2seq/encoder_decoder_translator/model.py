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
        tgt_vocab_size    = config.target_vocab_size
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

        logits = logits[:-1]                 # why'd i do this?
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
        return self.saver.save(self.sess, path, global_step=self.global_step)


    def predict_on_batch(self, x_batch, y_batch, l_batch):
        """ predict translation for a batch of inputs

            TODO - only take x_batch, and feed in the start symbol instead of
                    the first word from y (which is start symbol)
        """
        if not self.testing:
            print "WARNING: model not in testing mode. previous outputs won't be fed back into the decoder. Reconsider"

        probs = self.sess.run(self.output_probs, feed_dict={
                                    self.source: x_batch,
                                    self.target: y_batch,
                                    self.target_len: l_batch
                                })

        return probs
#        print probs[0]
#        return np.argmax(probs, axis=2)

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







class Seq2SeqV2(object):
    # DOESN'T WORK!!!!!
    def __init__(self, config, batch_size, testing=False, model_path=None):
        self.src_vocab_size    = config.src_vocab_size
        self.max_source_len    = config.max_source_len
        self.embedding_size    = config.embedding_size
        self.hidden_size       = config.hidden_size
        self.num_layers        = config.num_layers
        self.target_vocab_size    = config.target_vocab_size
        self.max_target_len    = config.max_target_len
        self.learning_rate    = config.learning_rate
        self.batch_size       = batch_size


        self.testing = testing


        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="source")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="target")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")
        # for counting interations
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


        self.build_variables()
        self.build_model()

        self.check = tf.add_check_numerics_ops()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def build_variables(self):
        #self.lr = tf.Variable(self.learning_rate, trainable=False, name="lr")
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("encoder1"):
            self.s_emb = tf.get_variable("s_embedding", shape=[self.src_vocab_size, self.embedding_size],
                                         initializer=initializer)
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[self.embedding_size, self.hidden_size],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.t_emb = tf.get_variable("t_embedding", shape=[self.target_vocab_size, self.embedding_size],
                                         initializer=initializer)
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.embedding_size, self.hidden_size],
                                            initializer=initializer)
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

            # projection
            self.proj_W = tf.get_variable("W", shape=[self.hidden_size, self.embedding_size],
                                          initializer=initializer)
            self.proj_b = tf.get_variable("b", shape=[self.embedding_size],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.embedding_size, self.target_vocab_size],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[self.target_vocab_size],
                                           initializer=initializer)

            # attention
            self.v_a = tf.get_variable("v_a", shape=[self.hidden_size, 1],
                                       initializer=initializer)
            self.W_a = tf.get_variable("W_a", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_a = tf.get_variable("b_a", shape=[self.hidden_size],
                                       initializer=initializer)
            self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                                       initializer=initializer)

    def build_model(self):
        with tf.variable_scope("encoder2"):
            # look up embedding
            source_x = tf.nn.embedding_lookup(self.s_emb, self.source)
            source_x = tf.unstack(source_x, axis=1)                         # split into a list of embeddings, 1 per word

            # run encoder over source sentence
            s = self.encoder.zero_state(self.batch_size, tf.float32)
            encoder_hs = []
            for t in range(self.max_source_len):
                if t > 0: tf.get_variable_scope().reuse_variables()         # let tf reuse variables
                x = source_x[t]
                projection = tf.matmul(x, self.s_proj_W) + self.s_proj_b    # project embedding into rnn's space
                h, s = self.encoder(projection, s)
                encoder_hs.append(h)
            encoder_hs = tf.pack(encoder_hs)

        with tf.variable_scope("decoder"):
            target_xs = tf.nn.embedding_lookup(self.t_emb, self.target)
            target_xs = tf.split(1, self.max_source_len, target_xs)

        # s = self.encoder.zero_state(self.batch_size, tf.float32)
        # encoder_hs = []
        # with tf.variable_scope("encoder"):
        #     for t in xrange(self.max_source_len):
        #         if t > 0: tf.get_variable_scope().reuse_variables()
        #         x = tf.squeeze(source_xs[t], [1])
        #         x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
        #         h, s = self.encoder(x, s)
        #         encoder_hs.append(h)
        # encoder_hs = tf.pack(encoder_hs)

        s = self.decoder.zero_state(self.batch_size, tf.float32)
        logits = []
        probs  = []
        with tf.variable_scope("decoder"):
            for t in xrange(self.max_source_len):
                if t > 0: tf.get_variable_scope().reuse_variables()
                if not self.testing or t == 0:
                    x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                h_t, s = self.decoder(x, s)
                h_tld = self.attention(h_t, encoder_hs)

                oemb  = tf.matmul(h_tld, self.proj_W) + self.proj_b
                logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
                prob  = tf.nn.softmax(logit)
                logits.append(logit)
                probs.append(prob)
                if self.testing:
                    x = tf.cast(tf.argmax(prob, 1), tf.int32)
                    x = tf.nn.embedding_lookup(self.t_emb, x)

        logits     = logits[:-1]
        targets    = tf.split(1, self.max_source_len, self.target)[1:]

        target_mask = tf.sequence_mask(self.target_len - 1, self.max_target_len - 1, dtype=tf.float32)
        loss_weights = tf.unstack(target_mask, None, 1)                  # 0/1 weighting for variable len tgt seqs
        self.loss = tf.nn.seq2seq.sequence_loss(logits, targets, loss_weights)

        self.probs = tf.transpose(tf.pack(probs), [1, 0, 2])

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)
#        gvs = optimizer.compute_gradients(self.loss)
#        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#        self.train_step = optimizer.apply_gradients(capped_gvs)


    def attention(self, h_t, encoder_hs):
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_source_len, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.mul(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.batch_matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), self.W_c) + self.b_c)

        return h_tld



    def predict_on_batch(self, x_batch, y_batch, l_batch):
        """ predict translation for a batch of inputs

            TODO - only take x_batch, and feed in the start symbol instead of
                    the first word from y (which is start symbol)
        """
        if not self.testing:
            print "WARNING: model not in testing mode. previous outputs won't be fed back into the decoder. Reconsider"

        probs = self.sess.run(self.output_probs, feed_dict={
                                    self.source: x_batch,
                                    self.target: y_batch,
                                    self.target_len: l_batch,
                                    self.dropout: 0.5
                                })

        return probs


    def train_on_batch(self, x_batch, y_batch, l_batch):
        """ train on a minibatch of data. x and y are assumed to be 
            padded to length max_seq_len, with l reflecting the original
            lengths of the target
        """
        _, loss, check = self.sess.run([self.train_step, self.loss, self.check],
                                feed_dict={
                                    self.source: x_batch,
                                    self.target: y_batch,
                                    self.target_len: l_batch,
                                    self.dropout: 1
                                })
        return loss, check

































class Seq2SeqV3(object):
    def __init__(self, config, batch_size, testing=False, model_path=None):
        self.src_vocab_size    = config.src_vocab_size
        self.max_source_len    = config.max_source_len
        self.embedding_size    = config.embedding_size
        self.hidden_size       = config.hidden_size
        self.num_layers        = config.num_layers
        self.target_vocab_size    = config.target_vocab_size
        self.max_target_len    = config.max_target_len
        self.learning_rate    = config.learning_rate
        self.batch_size       = batch_size


        self.testing = testing


        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="source")
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name="source_len")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="target")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")
        # for counting interations
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


        source_embedding = tf.get_variable('source_embeddings',
                                                shape=[self.src_vocab_size, self.embedding_size])
        target_embedding = tf.get_variable('target_embeddigns',
                                                shape=[self.target_vocab_size, self.embedding_size])

        source_embedded = tf.nn.embedding_lookup(source_embedding, self.source)
        # TODO IF INFERENCE, DO STUFF HERE
        target_embedded = tf.nn.embedding_lookup(target_embedding, self.target)

        self.decoder_output = self.encode_decode(source=source_embedded,
                                            source_len=self.source_len,
                                            target=target_embedded,
                                            target_len=self.target_len)
        # average per-word loss for each batch
        self.loss = self.cross_entropy_sequence_loss(logits=self.decoder_output,
                                                targets=self.target,
                                                seq_len=self.target_len)

        self.train_step = self.backward_pass(self.loss)

        self.check = tf.add_check_numerics_ops()

        self.sess = tf.Session()
#        self.writer =  tf.summary.FileWriter('./graphs', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
#       self.writer.close()




    def train_on_batch(self, x_batch, x_lens, y_batch, y_lens):
        """ train on a minibatch of data. x and y are assumed to be 
            padded to length max_seq_len, with l reflecting the original
            lengths of the target
        """
        _, logits, loss = self.sess.run([self.train_step, self.decoder_output, self.loss],
                                feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.target: y_batch,
                                    self.target_len: y_lens,
                                    self.dropout: 0.5
                                })

        return np.argmax(logits, axis=2), loss#np.mean(loss[loss > 0])


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



    def backward_pass(self, loss):
        train_step = tf.contrib.layers.optimize_loss(self.loss, None,
                self.learning_rate, "SGD", clip_gradients=5.0)

#        optimizer = tf.train.AdamOptimizer(self.learning_rate)
#        train_step = optimizer.minimize(loss)
        return train_step


    def cross_entropy_sequence_loss(self, logits, targets, seq_len):
        print logits
        print targets
        print seq_len
        # dillon's loss function
        logits     = tf.unstack(logits, axis=1)[:-1]
        targets    = tf.unstack(targets, axis=1)[1:]
        target_mask = tf.sequence_mask(seq_len - 1, self.max_target_len - 1, dtype=tf.float32)
        loss_weights = tf.unstack(target_mask, None, 1)                  # 0/1 weighting for variable len tgt seqs
        loss = tf.nn.seq2seq.sequence_loss(logits, targets, loss_weights)
        return loss

        # denny's loss function
        # logits = tf.transpose(logits, [1, 0, 2])
        # targets = tf.transpose(targets, [1, 0])

        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits,
        #     labels=targets)

        # # Mask out the losses we don't care about
        # loss_mask = tf.sequence_mask(
        #     tf.to_int32(seq_len), tf.to_int32(tf.shape(targets)[0]))
        # losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

        # return losses

        # my loss function
        # targets    = targets[:,1:]            # shift targets forward 1 space
        # logits     = logits[:,:-1,:]          # remove final token from logits so dimensions agree
        # seq_len    = seq_len - 1              # we've shortened each sequence by 1

        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits,
        #     labels=targets)

        # mask = tf.sequence_mask(seq_len, self.max_target_len-1, dtype=tf.float32)
        # losses = losses * mask



        # return losses


    def encode_decode(self, source, source_len, target, target_len):
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
        target_embedding = tf.get_variable("target_embedding",
                                           shape=[self.target_vocab_size, self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
        target_proj_W = tf.get_variable("t_proj_W",
                                        shape=[self.embedding_size, self.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        target_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())

        # projection to output
        out_embed_W = tf.get_variable("o_embed_W",
                                      shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        out_embed_b = tf.get_variable("o_embed_b",
                                      shape=[self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
        out_W = tf.get_variable("Wo", shape=[self.embedding_size, self.target_vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())
        out_b = tf.get_variable("bo", shape=[self.target_vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())

        target_x = tf.unstack(target, axis=1)

        # decode source by initializing with encoder's final hidden state
        s = initial_state
        logits = []
        for t in range(self.max_target_len):
            if t > 0: tf.get_variable_scope().reuse_variables()      # reuse variables after 1st iteration
            if self.testing and t == 0:
                pass # TODO EMBED SEQUENCE START TOKEN AND STICK IT INTO S
            elif not self.testing:
                x = target_x[t]

            projection = tf.matmul(x, target_proj_W) + target_proj_b # project embedding into rnn space
            h, s = decoder(projection, s)                            # s is last encoder state when t == 0
            
            out_embedding = tf.matmul(h, out_embed_W) + out_embed_b  # project output to target embedding space
            logit = tf.matmul(out_embedding, out_W) + out_b 
            logits.append(logit)

            if self.testing:
                x = tf.cast(tf.argmax(prob, 1), tf.int32)
                x = tf.nn.embedding_lookup(target_embedding, x)

        logits = tf.stack(logits)
        logits = tf.transpose(logits, [1, 0, 2])
        return logits


    def run_encoder(self, source, source_len, cell):
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=source,
            sequence_length=source_len,
            dtype=tf.float32)
        return {'outputs': outputs, 'final_state': state}


    def build_rnn_cell(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                    input_keep_prob=self.dropout,
                                                    output_keep_prob=self.dropout)
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
        return stacked_cell







