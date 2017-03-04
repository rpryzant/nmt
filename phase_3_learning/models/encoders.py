
import tensorflow as tf
from graph_module import GraphModule



class DefaultEncoder(GraphModule):
    def __init__(self, cell, name='default_encoder'):
        super(DefaultEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)
        return outputs, final_state


class DefaultBidirectionalEncoder(GraphModule):
    def __init__(self, cell, name='default_bidirectional'):
        super(DefaultBidirectionalEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs_pre, final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell,
            cell_bw=self.cell,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)
        # Concatenate outputs and states of the forward and backward RNNs
        outputs = tf.concat(2, outputs_pre)

        return outputs, final_state


class StackedBidirectionalEncoder(GraphModule):
    def __init__(self, cell, name='default_bidirectional'):
        super(StackedBidirectionalEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.python.ops.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=self.cell._cells,
                cells_bw=self.cell._cells,
                inputs=inputs,
                sequence_length=lengths,
                dtype=tf.float32)
        final_state = output_state_fw, output_state_bw

        return outputs, final_state


class HandmadeEncoder(GraphModule):
    def __init__(self, cell, embedding_size, hidden_size, max_source_len, batch_size, name='handmade_encoder'):
        super(HandmadeEncoder, self).__init__(name)
        self.cell = cell
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_source_len = max_source_len
        self.batch_size = batch_size    # TODO - PULL THIS FROM TENSOR SHAPE GET RID OF PARAM


    def _build(self, inputs, lengths):
        proj_W = tf.get_variable("s_proj_W", shape=[self.embedding_size, self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        proj_b = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        s = self.cell.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        for t in xrange(self.max_source_len):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = inputs[:,t]
            x = tf.matmul(x, proj_W) + proj_b
            h, s = self.cell(x, s)
            encoder_hs.append(h)
        encoder_hs = tf.pack(encoder_hs)
        outputs = tf.transpose(encoder_hs, [1, 0, 2])  # get into same shape as other encoder outputs
        final_state = s

        return outputs, final_state


class HandmadeBidirectionalEncoder(GraphModule):
    # TODO - try sticking two of the above handmades together? lots of repeated code

    def __init__(self, cell, embedding_size, hidden_size, max_source_len, batch_size, name='handmade_encoder'):
        super(HandmadeBidirectionalEncoder, self).__init__(name)
        self.cell = cell
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_source_len = max_source_len
        self.batch_size = batch_size


    def _build(self, inputs, lengths):
        # use same cell for forward and backward
        proj_Wf = tf.get_variable("s_proj_W", shape=[self.embedding_size, self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        proj_bf = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        s = self.cell.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        for t in xrange(self.max_source_len):
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = inputs[:,t]
            x = tf.matmul(x, proj_Wf) + proj_bf
            h, s = self.cell(x, s)
            encoder_hs.append(h)
        encoder_hs = tf.pack(encoder_hs)
        outputs_f = tf.transpose(encoder_hs, [1, 0, 2])  # get into same shape as other encoder outputs
        final_state_f = s

        proj_Wb = tf.get_variable("s_proj_W", shape=[self.embedding_size, self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        proj_bb = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        s = self.cell.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        for t in range(self.max_source_len)[::-1]:
            if t > 0: tf.get_variable_scope().reuse_variables()
            x = inputs[:,t]
            x = tf.matmul(x, proj_Wb) + proj_bb
            h, s = self.cell(x, s)
            encoder_hs.append(h)
        encoder_hs = tf.pack(encoder_hs)
        outputs_b = tf.transpose(encoder_hs, [1, 0, 2])  # get into same shape as other encoder outputs
        final_state_b = s

        outputs = tf.concat(2, [outputs_f, outputs_b])
        final_state = final_state_f, final_state_b
        
        return outputs, final_state


