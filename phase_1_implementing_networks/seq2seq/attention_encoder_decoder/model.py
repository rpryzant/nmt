"""
based on http://arxiv.org/abs/1412.7449 and https://arxiv.org/abs/1508.04025 (mostly the latter)


FINISHED READING DILLIONS CODE. LOOKS GOOD. UNDERSTAND PRETTY MUCH EVERYTHING EXCEPT
FOR A FEW TF MATRIX MANIPULATIONS. EASY ENOUGH TO FIGURE OUT WHEN YOU'RE PLAYING
WIT THE PLAYDOUGH. GONNA START THIS!!



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





class Seq2Seq:
    def __init__(self):
        
        self.source = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.target = 


        # build encoder
        with tf.variable_scope("encoder"):
            

