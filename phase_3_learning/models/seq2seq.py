"""
http://arxiv.org/abs/1412.7449 and https://arxiv.org/abs/1508.04025 (mostly the latter)


TODO: read from model path if provided
      beam search decoding
      unk replacement from attention scores
"""
import encoders
import decoders
import tensorflow as tf
import numpy as np
import os




class BaseModel():
    def save(self, path):
        raise NotImplementedError("Override me")

    def load(self, ckpt_dir):
        raise NotImplementedError("Override me")

    def train_on_batch(self, x_batch, x_lens, y_batch, y_lens, learning_rate=1.0):
        raise NotImplementedError("Override me")

    def run_on_batch(self, x_batch, x_lens, y_batch, y_lens, learning_rate=1.0):
        raise NotImplementedError("Override me")




class Seq2SeqV3(object):
    def __init__(self, config, dataset, sess, testing=False):
        self.sess                 = sess

        self.src_vocab_size       = config.src_vocab_size
        self.target_vocab_size    = config.target_vocab_size
        self.max_source_len       = config.max_source_len
        self.max_target_len       = config.max_target_len

        self.embedding_size       = config.embedding_size
        self.num_layers           = config.num_layers
        self.attention            = config.attention
        self.encoder_type         = config.encoder_type
        self.decoder_type         = config.decoder_type
        self.hidden_size          = config.hidden_size

        self.optimizer            = config.optimizer
        self.batch_size           = config.batch_size
        self.train_dropout        = config.dropout_rate
        self.max_grad_norm        = config.max_grad_norm

        self.dataset              = dataset
        self.testing              = testing

        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="lr")
        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_source_len], name="source")
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name="source_len")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_target_len], name="target")
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
#        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#        gpu_options = tf.GPUOptions(allow_growth=True)
#        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self.saver = tf.train.Saver()


    def save(self, path):
        self.saver.save(self.sess, path, global_step=self.global_step)


    def load(self, filepath=None, dir=None):
        print("\t Reading checkpoints...")
        if dir is not None:
            ckpt = tf.train.get_checkpoint_state(dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("\t No checkpoint found")
        elif filepath is not None:
            self.saver.restore(self.sess, filepath)
        else:
            raise Exception('\tERROR: must provide a checkpoint filepath or directory')


    def get_embeddings(self, source, target):
        """ source: [batch size, max length]    = source-side one-hot vectors
            target: [batch size, max length]    = target-side one-hot vectors
            returns word embeddings for each item in the source/target sequence
        """
        source_embedding = tf.get_variable('source_embeddings',
                                           shape=[self.src_vocab_size, self.embedding_size])
        source_embedded = tf.nn.embedding_lookup(source_embedding, source)
        self.target_embedding = tf.get_variable('target_embeddings',      
                                           shape=[self.target_vocab_size, self.embedding_size])
        target_embedded = tf.nn.embedding_lookup(self.target_embedding, target)

        return source_embedded, target_embedded

    def backward_pass(self, loss):
        """ use the given loss to construct a training step 
            NOTE: Using SGD instead of adagrad or adam because those don't seem to work
        """
        train_step = tf.contrib.layers.optimize_loss(self.loss, 
                                                      self.global_step,
                                                      learning_rate=self.learning_rate, 
                                                      optimizer=self.optimizer, 
                                                      clip_gradients=5.0)
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
            outputs, final_state = self.run_encoder(source, source_len, encoder_cell)

        with tf.variable_scope('decoder'):
            decoder_cell = self.build_rnn_cell()
            decoder_output = self.run_decoder(target, 
                                              target_len, 
                                              decoder_cell, 
                                              final_state if self.attention == 'off' else outputs)
        return decoder_output


    def run_encoder(self, source, source_len, cell):
        """ source: [batch size, seq len]
            source_len: [batch size]   = pre-padding lengths
            cell: RNNCell

            runs the cell inputs for source-len timesteps
        """

        if self.encoder_type == 'default':
            encoder = encoders.DefaultEncoder(cell)
        elif self.encoder_type == 'bidirectional' and self.num_layers == 1:
            encoder = encoders.DefaultBidirectionalEncoder(cell)
        elif self.encoder_type == 'bidirectional' and self.num_layers > 1:
            encoder = encoders.StackedBidirectionalEncoder(cell)
        elif self.encoder_type == 'handmade':
            encoder = encoders.HandmadeEncoder(cell,
                                               self.embedding_size,
                                               self.hidden_size,
                                               self.max_source_len,
                                               self.batch_size)
        elif self.encoder_type == 'handmade_bidirectional':
            encoder = encoders.HandmadeBidirectionalEncoder(cell,
                                                            self.embedding_size,
                                                            self.hidden_size,
                                                            self.max_source_len,
                                                            self.batch_size)
        outputs, final_state = encoder(source, source_len)
        return outputs, final_state


    def run_decoder(self, target, target_len, cell, encoder_result):
        """ target: [batch size, sequence len]
            target_len: [batch_size]  = pre-padded target lengths
            cell: RNNCell 
            encoder_result: 
                [batch_size, max_time, hidden state size] (encoder outputs)  if attention
                [batch size, hidden state size]       (encoder final state)  otherwisea

            runs a decoder on target and returns its predictions at each timestep
        """
        if self.decoder_type == 'argmax':
            decoder = decoders.ArgmaxDecoder(cell,
                                             self.embedding_size,
                                             self.hidden_size,
                                             self.target_vocab_size,
                                             self.batch_size,
                                             self.max_target_len,
                                             self.target_embedding,
                                             self.testing,
                                             self.attention,
                                             self.encoder_type)
        logits = decoder(target, target_len, encoder_result)
        return logits


    def build_rnn_cell(self):
        """ builds a stacked RNNCell according to specs defined in the model's config
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
#                                             input_keep_prob=(1-self.dropout),
                                             output_keep_prob=(1-self.dropout))
        stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
        return stacked_cell


    def train_on_batch(self, x_batch, x_lens, y_batch, y_lens, learning_rate=1.0):
        """ train on a minibatch of data. x and y are assumed to be 
            padded to length max_seq_len, with [x/y]_lens reflecting
            the original lengths of each sequence
        """
        _, logits, loss, step = self.sess.run([self.train_step, self.decoder_output, self.loss, self.global_step],
                                feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.target: y_batch,
                                    self.target_len: y_lens,
                                    self.dropout: self.train_dropout,
                                    self.learning_rate: learning_rate
                                })
        return np.argmax(logits, axis=2), loss, step


    def run_on_batch(self, x_batch, x_lens, y_batch, y_lens, learning_rate=1.0):
        """ "predict" on a batch while the model is in training mode (for validation purposes)
        """
        logits, loss = self.sess.run([self.decoder_output, self.loss],
                                feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.target: y_batch,
                                    self.target_len: y_lens,
                                    self.dropout: 0.0,
                                    self.learning_rate: learning_rate
                                })

        return np.argmax(logits, axis=2), loss



    def predict_on_batch(self, x_batch, x_lens):
        """ predict translation for a batch of inputs. for testing mode only.
        """
        assert self.testing, 'ERROR: model must be in test mode to make predictions!'

        logits = self.sess.run(self.decoder_output, feed_dict={
                                    self.source: x_batch,
                                    self.source_len: x_lens,
                                    self.dropout: 0.0,
                                })

        return np.argmax(logits, axis=2)









