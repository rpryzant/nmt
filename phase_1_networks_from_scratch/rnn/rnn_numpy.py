"""
Implementing an rnn from scratch with numpy

This rnn is intended to be used in text domains. So inputs are one-hot encoded,
   and a softmax over an output distribution is used to select maximum likelihood predictions
"""


class RNN(object):
    
    def __init__(self, input_dim, hidden_dim, bptt_clip=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_clip = bptt_clip

        # initialize weights on uniform [-1/sqrt(input), 1/sqrt(input)] 
        #  a la http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        input_lim = np.sqrt(1.0 / self.input_dim)
        hidden_lim = np.sqrt(1.0 / self.hidden_dim)

        # encoding/embeddings of input. Each column i of U is a vector encoding of x_i. 
        # Operations like U[:,x[i]], then, act like multiplying U with a one-hot vector to pull out that vector's embedding
        self.U = np.random.uniform(-input_lim, input_lim, (self.hidden_dim, self.input_dim))
        # output cell weights
        self.V = np.random.uniform(-hidden_lim, hidden_lim, (self.input_dim, self.hidden_dim))
        # recurrent weights
        self.W = np.random.uniform(-hidden_lim, hidden_lim, (self.hidden_dim, self.hidden_dim))

    def softmax(self, x):
        sf = np.exp(x)
        sf = sf/np.sum(sf, axis=0)
        return sf

    def forward_pass(self, x):
        """ network forward pass """
        T = len(x) # num timesteps

        # remember all hidden states and outputs at each timestep (for bptt). 
        # add extra 0 hidden state to get things going 
        h = np.zeros((T + 1, self.hidden_dim))
        o = np.zeros((T, self.input_dim))

        for t in range(T):
            # s_t = tanh(U * x_t   +    W *  s_{t-1})
            s[t] = np.tanh(self.U[:,x[t]] + np.dot(self.W, s[t-1]))
            # o_t = softmax(V * s_t)
            o[t] = softmax(np.dot(self.V, s[t]))

        return o, s

    def predict(self, x):
        """ forward pass to get probabilities, then select attribute with highest probability """
        o, s = self.forward_pass(x)
        return np.argmax(o, axis=1) # return the argmax of each output distribution


    def cumulative_cross_entropy_loss(self, X, Y):
        """ cross entropy loss across a whole dataset
              L(Y, O) = -(1/N) sum y_n log o_n
        """
        L = 0
        for i in range(len(Y)):
            o, s = self.forward_pass(X[i])
            # get outputted probabilities for each true label
            predictions_for_labels = o[range(len(Y[i])), Y[i]] 
            L += -np.sum(np.log(predictions_for_labels))
    
        # normalize by num training examples
        N = np.sum(len(y) for y in Y)

        return L / N






