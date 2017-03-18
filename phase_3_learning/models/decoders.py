"""
TODO

srructure as

Decoder  (collects args, builds decoder vars, hold attention functions)
   TrainDecoder

   ArgmaxTestDecoder

   BeamSearchTestDecoder
  
  T


"""
import numpy as np
import tensorflow as tf
from graph_module import GraphModule
import sys; sys.path.append('../')    # sigh 
from msc.constants import Constants



class ArgmaxDecoder(GraphModule):
    def __init__(self, cell, embedding_size, hidden_size, target_vocab_size, 
                 batch_size, max_target_len, target_embedding, testing, attention, 
                 encoder_type, name='argmax_decoder'):
        super(ArgmaxDecoder, self).__init__(name)        
        self.cell = cell
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.max_target_len = max_target_len
        self.target_embedding = target_embedding
        self.testing = testing
        self.attention = attention
        self.encoder_type = encoder_type

    def _build(self, target, target_len, encoder_result):
        """ target: [batch size, sequence len]
            target_len: [batch_size]  = pre-padded target lengths
            decoder: RNNCell 
            encoder_result: 
                [batch_size, max_time, hidden state size] (encoder outputs)  if attention
                [batch size, hidden state size]       (encoder final state)  otherwisea

            runs a decoder on target and returns its predictions at each timestep
        """
        self.build_decoder_vars()
        # decode source by initializing with encoder's final hidden state
        target_x = tf.unstack(target, axis=1)                        # break the target into a list of word vectors
   
        if self.attention == 'off':
            s = encoder_result 
        else:
            s = self.cell.zero_state(self.batch_size, tf.float32)
            self.build_attention_vars()

        self.attention_probs = []                 # for viz, gather attention probs
        logits = []
        for t in range(self.max_target_len):
            if t > 0: tf.get_variable_scope().reuse_variables()      # reuse variables after 1st iteration
            if self.testing and t == 0:                               # at test time, kick things off w/start token
                start_tok = Constants.START_I                        # get start token index for decoding lang
                start_vec = np.full([self.batch_size], start_tok)
                x = tf.nn.embedding_lookup(self.target_embedding, start_vec)   # look up start token
            elif not self.testing:
                x = target_x[t]                                      # while training, feed in correct input

            projection = tf.matmul(x, self.target_proj_W) + self.target_proj_b # project embedding into rnn space
            h, s = self.cell(projection, s)                            # s is last encoder state when t == 0

            if not self.attention == 'off':
                h, a_t = self.attention_layer(h, encoder_result)
            
            self.attention_probs.append(a_t)
            out_embedding = tf.matmul(h, self.out_embed_W) + self.out_embed_b  # project output to target embedding space
            logit = tf.matmul(out_embedding, self.out_W) + self.out_b 
            logits.append(logit)

            if self.testing:
                prob = tf.nn.softmax(logit)
                x = tf.cast(tf.argmax(prob, 1), tf.int32)            # if testing, use cur pred as next input
                x = tf.nn.embedding_lookup(self.target_embedding, x)

        logits = tf.stack(logits)
        logits = tf.transpose(logits, [1, 0, 2])                     # get into [batch size, sequence len, vocab size]
        return logits
        

    def attention_layer(self, h_t, encoder_states):
        """ dot-product attentional layer as described in https://arxiv.org/abs/1508.04025
            -h_t : [batch size, hidden size]   : current decoder hidden state
            -encoder_states [batch size, max len, hidden size] : history of encoder activity
            - W_a, W_c, b_c : weights for attention
        """
        def dot_score(encoder_states, h_t):  # returns [50, 5]
            """ dots h_t with each encoder state """
            encoder_states = tf.transpose(encoder_states, [1, 0, 2]) 
            return tf.reduce_sum(tf.mul(encoder_states, h_t), 2)     

        def bilinear_score(encoder_states, h_t):
            """ h_t * W_a, then dots the result with each encoder state """
            encoder_states = tf.transpose(encoder_states, [1, 0, 2]) 
            pre_score = tf.matmul(h_t, self.W_a) 
            return tf.reduce_sum(tf.multiply(encoder_states, pre_score), 2)

        # compute attentional weighting
        if self.attention == 'dot':
            scores = dot_score(encoder_states, h_t)
        elif self.attention == 'bilinear':
            scores = bilinear_score(encoder_states, h_t)
        else:
            raise RuntimeException('ERROR: attention type %s not supported' % self.attention)

        a_t    = tf.nn.softmax(scores, dim=0)                      # softmax over timesteps for each batch
        a_t    = tf.expand_dims(a_t, 2)                            # turn into 3d tensor

        #          ( encoder )      ( a_t )
        # perform [hidden, time] X [time, 1] slice matmuls to get weighted average of 
        # encoder states, weighted by a_t
        a_t = tf.transpose(a_t, [1, 0, 2])                           # move batch to 1st dim, time to 2nd dim
        encoder_states = tf.transpose(encoder_states, [0, 2, 1])     # move batch to 1st dim, time to last dim
        c_t    = tf.batch_matmul(encoder_states, a_t)  
        c_t    = tf.squeeze(c_t)

        # concat h_t and c_t, then send through fc layer to get final h
        h_new  = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), self.W_c) + self.b_c)
        return h_new, a_t


    def build_decoder_vars(self):
        """ builds all the matrices needed for decoding
        """
        # projection to rnn space
        self.target_proj_W = tf.get_variable("t_proj_W",
                                             shape=[self.embedding_size, self.hidden_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
        self.target_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
        # projection to output embeddings
        self.out_embed_W = tf.get_variable("o_embed_W",
                                           shape=[self.hidden_size, self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
        self.out_embed_b = tf.get_variable("o_embed_b",
                                           shape=[self.embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
        # projection to logits
        self.out_W = tf.get_variable("Wo", shape=[self.embedding_size, self.target_vocab_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
        self.out_b = tf.get_variable("bo", shape=[self.target_vocab_size],
                                     initializer=tf.contrib.layers.xavier_initializer())

        
    def build_attention_vars(self):
        """ builds all the matrices needed for attention
        """
        # if we were bidirectional, hidden states have been concatenated, so double the dimension
        extra_hidden_features = self.hidden_size if 'bidirectional' in self.encoder_type else 0
        self.W_a = tf.get_variable("W_a", shape=[self.hidden_size, self.hidden_size + extra_hidden_features],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size + extra_hidden_features, self.hidden_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                                   initializer=tf.contrib.layers.xavier_initializer())

        
