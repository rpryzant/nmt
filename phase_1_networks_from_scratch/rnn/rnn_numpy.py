"""
Implementing an rnn from scratch with numpy

This rnn is intended to be used in text domains. So inputs are one-hot encoded,
   and a softmax over an output distribution is used to select maximum likelihood predictions

This rnn is 2 layers. The first layer is an embedding, and the second is a standard recursive cell
"""
import sys
import numpy as np


VOCABULARY_SIZE = 8000 # from generate_data


class RNN(object):
    
    def __init__(self, input_dim, hidden_dim=100, bptt_clip=4):
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
            h[t] = np.tanh(self.U[:,x[t]] + np.dot(self.W, h[t-1]))
            # o_t = softmax(V * s_t)
            o[t] = self.softmax(np.dot(self.V, h[t]))

        return o, h

    def predict(self, x):
        """ forward pass to get probabilities, then select attribute with highest probability """
        o, _ = self.forward_pass(x)
        return np.argmax(o, axis=1) # return the argmax of each output distribution


    def backward_pass(self, x, y):
        """ backpropagation through time (bptt) """
        T = len(y)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        o, h = self.forward_pass(x)
        
        do = o
        do[range(len(y)), y] -= 1.0 # subtract 1 from each true label

        # step backwards through time
        for t in range(T)[::-1]:
            # dL/dV = dL/do * hidden output
            dV += np.outer(do[t], h[t].T)            
            # gradient to send back through time 
            #  initialized to    V^T * dL/do  *  derivative of tanh (hidden activation function)
            dt = np.dot(self.V.T, do[t]) * (1 - (h[t] ** 2))
            # propagate gradient back through time at most self.bptt_clip steps
            for step in range(max(0, t - self.bptt_clip), t + 1)[::-1]:
                # dL/dW = sum of (dL/dt * h_s^T   (i.e. elementwise product))'s for all time steps
                dW += np.outer(dt, h[step - 1])
                # dL/dU = sum of (dL/dt)'s, but only for the column that x_s corresponds to
                #    i.e pretend x_s is a one-hot vector
                dU[:,x[step]] += dt

                # prepare dL/dt for next time step
                dt = np.dot(self.W.T, dt) * (1 - h[step - 1] ** 2)

        return dU, dV, dW


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





# init model and data
rnn = RNN(VOCABULARY_SIZE)
X, Y = eval(open(sys.argv[1]).read())

print rnn.backward_pass(X[10], Y[10])



print '====== PRE-TRAINING SANITY CHECK'
# sanity check: before training, loss should be close to that of random predictions
print "\t Observed loss: ", rnn.cumulative_cross_entropy_loss(X[:500], Y[:500])
# with random predictions, each word is selected with probability 
#        1/|V|, so cross entropy woud be 
#   L = - (1/N) N log (1/|V|)
#     = log |V|
print "\t Expected loss: ", np.log(VOCABULARY_SIZE)

