"""
Implementation of the skip-gram model for generating word vectors

From "Efficient Estimation of Word Representations in Vector Space"
(https://arxiv.org/pdf/1301.3781.pdf)


"""
import numpy as np


class SkipGram(object):

    def __init__(self, vocab_size):
        self.input_vectors = self.__normalizeRows(np.random.randn(5, 3))    # word vectors you're learning
        self.output_vectors = self.__normalizeRows(np.random.randn(5, 3))   # output vectors
        self.vocab_size = vocab_size                                        # |V|


    def forward_pass(self, word_index):
        one_hot = np.zeros((1, self.vocab_size))
        one_hot[:, word_index] = 1
        
        # get vector representation of center word
        center_vec = np.dot(one_hot, self.input_vectors)

        # predict and return probability of all outputs being in the context
        scores = np.dot(center_vec, self.output_vectors.T)
        probs = self.__softmax(scores)

        return scores, probs


    def backward_pass(self, loss, scores, probs):
        



s = SkipGram()

corpus = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])




