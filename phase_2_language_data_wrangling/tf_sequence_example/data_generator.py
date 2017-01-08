

import tensorflow as tf
import sys

class DataGenerator(object):
    def __init__(self, TFRecords_loc, batch_size):  # TODO variable num inputs for multiple data files
        data_loc = TFRecords_loc
        filename_queue = tf.train.string_input_producer([TFRecords_loc])
        reader = tf.TFRecordReader()

        context_features = {
            'length': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'tokens': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        _, ex = reader.read(filename_queue)

        contex_parsed, seq_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        length = contex_parsed['length']
        tokens = seq_parsed['tokens']
        labels = seq_parsed['labels']

        self.batch = tf.train.batch(
            tensors=[tokens, labels, length],
            batch_size=batch_size,
            dynamic_pad=True,
        )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)


    def next_batch(self):
        batch = self.sess.run([self.batch])
        return batch[0][0], batch[0][1], batch[0][2]



#dg = DataGenerator(sys.argv[1], 3)


#print dg.next_batch()




# print dg.next_batch()
