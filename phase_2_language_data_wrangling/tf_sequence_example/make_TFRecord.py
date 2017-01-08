"""
This file takes in parsed data, and writes one or more TFRecord files.


"""
import tensorflow as tf
import numpy as np
import sys

def make_example(seq, labels):
    """ make a tf.SequenceExample from an x, y sequence pair
    """
    ex = tf.train.SequenceExample()

    # pack in length information
    seq_len = len(seq)
    ex.context.feature['length'].int64_list.value.append(seq_len)

    # pack in sequential features for each example
    seq_features = ex.feature_lists.feature_list['tokens']
    label_features = ex.feature_lists.feature_list['labels']
    for x_id, label_id in zip(seq, labels):
        seq_features.feature.add().int64_list.value.append(x_id)
        label_features.feature.add().int64_list.value.append(label_id)

    return ex


INPUT = sys.argv[1]
OUTPUT = sys.argv[2]


# read in raw data
# X and Y are paired lists of the form [ [id1, id2, ... ], ... ]
# X contains sequences, and Y contains labels
X, Y = eval(open(INPUT).read())

writer = tf.python_io.TFRecordWriter(OUTPUT)
for x, y in zip(X, Y):
    ex = make_example(x, y)
    writer.write(ex.SerializeToString())
writer.close()

print 'wrote to ', OUTPUT


