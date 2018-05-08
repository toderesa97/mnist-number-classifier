import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set

valid_x, valid_y = valid_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print(train_y[57])

# OK until here

# TODO: the neural net!!


train_y = one_hot(train_y, 10)  # the labels are in the last row. Then we encode them in one hot code



x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# HIDDEN LAYER
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# SALDIA DE LAS NEURONAS
W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20

training_errors = []
validation_errors = []
epoch = 0
while epoch < 100:
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    training_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    validation_error = sess.run(loss, feed_dict={x: valid_x, y_:one_hot(valid_y, 10)})
    validation_errors.append(validation_error)
    training_errors.append(training_error)
    print("Epoch #:", epoch, "Error: ", training_error)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
    epoch = epoch + 1
    if (len(validation_errors) >= 2 and abs(validation_errors[epoch-2] - validation_errors[epoch-1]) < 0.5):
        break

x_axis_training_errors = list(range(1, len(training_errors)+1))
plt.plot(x_axis_training_errors, training_errors)
plt.show()
plt.plot(x_axis_training_errors, validation_errors)
plt.show()