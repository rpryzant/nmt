

import tensorflow as tf
import sys

class DataGenerator(object):
    """ Object that is initialized with the location of a TFRecords file, and 
          can spit out batches of x, y pairs from that file
    """
    def __init__(self, TFRecords_loc, batch_size):  # TODO variable num inputs for multiple data files
        # initialize TFRecordReader and filename queue (important for when I extend this to multiple datapaths)
        data_loc = TFRecords_loc
        filename_queue = tf.train.string_input_producer([TFRecords_loc])  # could use multiple locations 
        reader = tf.TFRecordReader()

        # the expected protobuf format
        context_features = {
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        # op for reading a TFRecord from file
        _, ex = reader.read(filename_queue)

        # op for parsing that example
        contex_parsed, seq_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        # extract stuff you care about
        length = contex_parsed['length']
        tokens = seq_parsed['tokens']
        labels = seq_parsed['labels']

        # batching op. Note how dynamic_pad is set to **True**
        self.batch = tf.train.batch(
            tensors=[tokens, labels, length],
            batch_size=batch_size,
            dynamic_pad=True,
        )

        # get the session ready to go
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)


    def next_batch(self):
        """ Spits out a batches worth of data
        """
        # run the batch op
        batch = self.sess.run([self.batch])
        # return (tokens, labels, lengths)
        return batch[0][0], batch[0][1], batch[0][2]



#dg = DataGenerator(sys.argv[1], 3)


#print dg.next_batch()




# print dg.next_batch()
