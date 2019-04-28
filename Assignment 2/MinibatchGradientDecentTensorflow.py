import numpy as np
import scipy
import tensorflow as tf
import sys

# make log direction using current time
from datetime import datetime

# current time as string
ctime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, ctime)


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


def calculate_accuracy():

    testData = np.load('test_data.npy')
    testData = np.concatenate(
        (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)
    testLabel = np.load('test_data_label.npy')

    cnt_right_prediction = 0
    cnt_wrong_prediction = 0

    for idx, point in enumerate(testData):
        predicted_label = activation_funtion(np.matmul(point, weightvalues))
        actual_label = int(testLabel[idx])

        if actual_label == predicted_label:
            print("Right Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label, "for data : ", point)
            cnt_right_prediction = cnt_right_prediction + 1
        else:
            print("XXX !!! ", "Wrong Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label, "for data : ", point)
            cnt_wrong_prediction = cnt_wrong_prediction + 1

    accuracy = cnt_right_prediction/( cnt_right_prediction+cnt_wrong_prediction )
    print("Correct Prediction of the model from Test Dataset : ", cnt_right_prediction)
    print("Incorrect Prediction of the model from Test Dataset : ", cnt_wrong_prediction)
    return accuracy


try:
    xdata = np.load("train_data.npy")
    xdata = np.concatenate((xdata.reshape(xdata.shape[0], 2), np.ones(xdata.shape[0]).reshape(xdata.shape[0], 1)), axis=1)
    zdata = np.load("train_data_label.npy")

    N, p = xdata.shape

    X = tf.placeholder(tf.float32, shape=(None, p), name="X")  # no length yet
    z = tf.placeholder(tf.float32, shape=(None, 1), name="z")  # no length yet

    # minibatch gradient descent solution
    if len(sys.argv) == 1:
        batch_size = xdata.shape[0] // 10             # 10 batches
    else:
        batch_size = int(sys.argv[1])

    print(batch_size)

    eta = 0.1
    n_iterations = 500

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

    # execution
    with tf.Session() as session:
        session.run(init)
        for epoch in range(n_iterations):
            print(" --------->>>>", epoch)
            mini_batches = create_mini_batches(xdata, zdata.reshape(-1, 1), batch_size)
            for mini_batch in mini_batches:
                x_mini, z_mini = mini_batch
                session.run(training_step, feed_dict={X:x_mini, z:z_mini.reshape(-1, 1)})

                if epoch % 10 == 0:     # write summary
                    summary = session.run(mse_summary, feed_dict={X:x_mini, z:z_mini.reshape(-1, 1)})
                    step = epoch
                    file_writer.add_summary(summary, step)

        file_writer.close()
        weightvalues = weights.eval()

    accuracy = calculate_accuracy()
    print("")
    print("Calculated Weights : ", weightvalues)
    print("")
    print("Accuracy of the model (Minibatch Gradient Descent with Tensorflow ) : ", accuracy)

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])