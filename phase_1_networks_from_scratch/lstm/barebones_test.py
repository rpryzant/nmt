"""
My tf lstm had some weird behavior at first (losses bouncing around regaurdless of learning rate)
 so I'm going to spend some time stripping my implementation down until I get something that works.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

E = 15    # num batches
B = 2     # batch size
L = 5     # max seq len
H = 8     # num hidden units
C = 10    # num classes



# placeholders for data
input_placeholder = tf.placeholder(tf.int32, shape=[None, L])
target_placeholder = tf.placeholder(tf.int32, shape=[None, L])
length_placeholder = tf.placeholder(tf.int32, shape=[None])

# retrieve and unpack input embeddings
U = tf.get_variable(
    name="U",
    initializer=tf.random_normal_initializer(),
    shape=[C, H])
input = tf.nn.embedding_lookup(U, input_placeholder)


cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=H)
outputs, state = tf.nn.dynamic_rnn(
    cell=cell,
    inputs=input,
    sequence_length=length_placeholder,
    initial_state=cell.zero_state(B, tf.float32),
    dtype=tf.float32)

# output layer
V = tf.get_variable(
    name='V',
    initializer=tf.random_normal_initializer(),
    shape=[H, C])

outputs = tf.reshape(outputs, [-1, H])    # smush together all the batches so that we can calculate loss in one step
logits = tf.matmul(outputs, V)

# loss
target_flat = tf.reshape(target_placeholder, [-1])  # flatten out batches for same reason as above
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target_flat)

# mask out padded targets from loss
mask = tf.sign(tf.to_float(target_flat)) 
losses = mask * losses

# recover batches and compute mean loss per batch 
losses = tf.reshape(losses, tf.shape(target_placeholder))
loss_per_batch = tf.reduce_sum(losses, reduction_indices=1) / tf.to_float(length_placeholder)
mean_batch_loss = tf.reduce_mean(loss_per_batch)

# training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003)
train_step = optimizer.minimize(mean_batch_loss)







###########################3



example_lengths = np.random.randint(1, L, [E * B])

# start with random data, then do Y as shifted and insert start/end chars
X = np.random.randint(1, C, [E * B, L])
Y = np.random.randint(1, C, [E * B, L])

# make examples ragged
for i, length in enumerate(example_lengths):
    X[i, length:] = 0
    Y[i, length:] = 0

# batch up data
X_train = []
Y_train = []
train_lengths = []
for i in range((E * B) - B + 1)[::B]:
    X_train.append(X[i:i+B])
    Y_train.append(Y[i:i+B])
    train_lengths.append(example_lengths[i:i+B])
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
train_lengths = np.asarray(train_lengths)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(1000):
    epoch_loss = 0
    for batch_i in tqdm(range(len(X_train))):
        _, loss = sess.run([train_step, mean_batch_loss],
                           feed_dict={
                               input_placeholder: X_train[batch_i],
                               target_placeholder: Y_train[batch_i],
                               length_placeholder: train_lengths[batch_i]
                           })
        epoch_loss += loss
    print epoch_loss
                           

                           
