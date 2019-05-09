import numpy as np
import scipy
import tensorflow as tf
import sys

xdata = np.load("linearData.npy")
zdata = np.load("observed.npy")
try:
    N, p = xdata.shape
    X = tf.constant(xdata, dtype=tf.float32, name="X")
    z = tf.constant(zdata.reshape(-1, 1), dtype=tf.float32, name="z")

    # batch gradient descent solution
    eta = 0.1
    n_iterations = 500

    Xt = tf.transpose(X)
    weights = tf.Variable(tf.random_uniform([p, 1], -1.0, 1.0), name="weights")

    y = tf.matmul(X, weights, name="predictions")

    error = y - z

    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2 * tf.matmul(Xt, error) / N
    training_step = tf.assign(weights, weights - eta * gradients)

    init = tf.global_variables_initializer()

    # execution
    with tf.Session() as session:
        session.run(init)
        for epoch in range(n_iterations):
            session.run(training_step)

        weights = weights.eval()

    print(weights)
except:
    ex = sys.exc_info()
    print("EXception Raised in Training Phrase !!! ")
    print("Exception Details  :  ", ex[0])
