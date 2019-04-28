#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:12:57 2019

@author: c00303945
"""
import numpy as np
import scipy
import tensorflow as tf
import sys

# make log direction using current time
from datetime import datetime


def create_mini_batches(X, z, batch_size):
    mini_batches = []
    data = np.hstack((X, z))
    np.random.shuffle(data)
    n_minibatches = data.shape[0]               # batch_size

    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        z_mini = mini_batch[:, -1]              # z is a vector, not a 1-column array
        mini_batches.append((X_mini, z_mini))

        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            z_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, z_mini))
    return mini_batches


def activation_funtion(val):
    if val >= 0:
        return 1
    else:
        return -1

def miniBatchTensorflow_accuracy():

    testData = np.load('test_data.npy')
    newData = np.concatenate((testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)
    testLabel = np.load('test_data_label.npy')

    rightP_counter = 0
    wrongP_counter = 0

    for index, trueData in enumerate(newData):
        p_label = activation_funtion(np.matmul(trueData, weightvalues))
        t_label = testLabel[index]

        if abs(t_label-p_label) > 0:
            wrongP_counter = wrongP_counter + 1
            print("Miss Prediction: ", t_label, p_label)
        else:
            rightP_counter = rightP_counter+1
            print("Correct Prediction: ", t_label, p_label)

    accuracy = rightP_counter/( rightP_counter+wrongP_counter )
    return accuracy

# current time as string
ctime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, ctime)

#load input data
xindata = np.load("train_data.npy")
xindata = np.concatenate((xindata.reshape(xindata.shape[0], 2), np.ones(xindata.shape[0]).reshape(xindata.shape[0], 1)), axis=1)

labeldata = np.load("train_data_label.npy")

N, p = xindata.shape

X = tf.placeholder(tf.float32, shape=(None, p), name="X")  # no length yet
z = tf.placeholder(tf.float32, shape=(None, 1), name="z")  # no length yet

# minibatch gradient descent solution
if len(sys.argv) == 1:
    batch_size = xindata.shape[0]//10    # 10 batches
else:
    batch_size = int(sys.argv[1])

print(batch_size)

eta = 0.1
n_iterations = 50

Xt = tf.transpose(X)
weights = tf.Variable(tf.random_uniform([p, 1], -1.0, 1.0), name="weights")

y = tf.matmul(X, weights, name="predictions")

error = y - z

mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [weights])[0]

training_step = tf.assign(weights, weights - eta * gradients)

init = tf.global_variables_initializer()


# create a node that evaluates the mse value and write to binary log string
mse_summary = tf.summary.scalar('mse', mse)
# create logfile writer for summaries
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# for saving the session
saver = tf.train.Saver()

    # execution
with tf.Session() as session:
    session.run(init)

    for epoch in range(n_iterations):
        print("Epoch --> ", epoch)
        mini_batches = create_mini_batches(xindata, labeldata.reshape(-1,1), batch_size)
        for mini_batch in mini_batches:
            x_mini, z_mini = mini_batch
            session.run(training_step, feed_dict={X:x_mini, z:z_mini.reshape(-1,1)})
            if epoch % 5 == 0: # write summary
                summary = session.run(mse_summary, feed_dict={X:x_mini, z:z_mini.reshape(-1,1)})
                step = epoch
                file_writer.add_summary(summary, step)
                
            print("MSE : ", mse)

    weightvalues = weights.eval()
    file_writer.close()    


print(weightvalues)

accuracy = miniBatchTensorflow_accuracy()
print("\nAccuracy of the minibatch tensorflow : ", accuracy)


