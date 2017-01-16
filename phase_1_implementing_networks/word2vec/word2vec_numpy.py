"""
Implementation of the skip-gram model for generating word vectors

From "Efficient Estimation of Word Representations in Vector Space"
(https://arxiv.org/pdf/1301.3781.pdf)


"""
import numpy as np


class SkipGram(object):

    def __init__(self, vocab_size, learning_rate):
        self.input_vectors = self.__normalizeRows(np.random.randn(5, 3))    # word vectors you're learning
        self.output_vectors = self.__normalizeRows(np.random.randn(5, 3))   # output vectors
        self.vocab_size = vocab_size                                        # |V|
        self.learning_rate = learning_rate

    def forward_pass(self, word_index):
        one_hot = np.zeros((1, self.vocab_size))
        one_hot[:, word_index] = 1
        
        # get vector representation of center word
        center_vec = np.dot(one_hot, self.input_vectors)

        # predict and return probability of all outputs being in the context
        scores = np.dot(center_vec, self.output_vectors.T)
        probs = self.__softmax(scores)

        return scores, probs, center_vec


    def backward_pass(self, loss, target_i, scores, probs, center_vec):
        # softmax gradient
        grad_scores = probs
        grad_scores[:, target_i] -= 1      

        # gradient w/r/t input vector that corresponds to center word
        grad_center_vec = np.dot(grad_scores, self.output_vectors)
        
        # gradient w/r/t all output vectors
        grad_output_vecs = np.dot(grad_scores.T, center_vec)

        return grad_center_vec, grad_output_vecs


    def loss(self, probs, target_i):
        # cross entropy loss: - \sum y_i log yhat_i  = - log yhat_center
        return -np.log(probs[:, target_i])


    def train_step(self, current_word_i, context_words_i):
        loss = 0
        grad_input = np.zeros(self.input_vectors.shape)
        grad_output = np.zeros(self.output_vectors.shape)

        for word_i in context_words_i:
            scores, probs, center_vec = self.forward_pass(current_word_i)
            l = self.loss(probs, word_i)
            grad_center_vec, grad_output_vecs = self.backward_pass(l, word_i, scores, probs, center_vec)
            # accumulate gradients
            grad_input[current_word_i:current_word_i+1] += grad_center_vec
            grad_output += grad_output_vecs

        # sgd step
        self.input_vecs -= self.learning_rate * grad_input
        self.output_vecs -= self.learning_rate * grad_output
            

    def __softmax(self, x):
        if len(scores.shape) > 1:
            # matrix
            x = x.T - np.max(x, 1)               # subtract off max for numerical stability
            x = np.exp(x) / np.sum(np.exp(x), 0)
            x = x.T
        else:
            # vector
            x -= np.max(x)                       # again, stability
            x = np.exp(x) / np.sum(np.exp(x))

        return x

    def __normalizeRows(self, x):
        # normalizes each row of a matrix to have magnitude 1
        x_magnitude = np.sqrt(np.sum(x**2, axis=1))
        for i, row in enumerate(x):
            x[i] /= x_magnitude[i]

        return x




s = SkipGram()

corpus = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])




