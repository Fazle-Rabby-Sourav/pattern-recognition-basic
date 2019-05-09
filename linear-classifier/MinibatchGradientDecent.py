import numpy as np
import scipy
import sys


# linear regression using "mini-batch" gradient descent
# function to create a list containing mini-batches
def create_mini_batches(X, z, batch_size):
    mini_batches = []
    data = np.hstack((X, z))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        z_mini = mini_batch[:, -1] # z is a vector, not a 1-column array
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
    testLabel = np.load('test_data_label.npy')

    tdata = np.concatenate(
        (testData.reshape(testData.shape[0], 2), np.ones(testData.shape[0]).reshape(testData.shape[0], 1)), axis=1)

    cnt_right_prediction = 0
    cnt_wrong_prediction = 0

    for idx, point in enumerate(tdata):
        predicted_label = activation_funtion(np.matmul(point, w))
        actual_label = int(testLabel[idx])

        if actual_label == predicted_label:
            print("Right Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label)
            cnt_right_prediction = cnt_right_prediction + 1
        else:
            print("Wrong Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label)
            cnt_wrong_prediction = cnt_wrong_prediction + 1

    accuracy = cnt_right_prediction/( cnt_right_prediction+cnt_wrong_prediction )
    return accuracy


try:
    # input
    x = np.load('train_data.npy')

    # observed values
    z = np.load('train_data_label.npy')

    # make data matrix
    xdata = np.concatenate((x.reshape(x.shape[0], 2), np.ones(x.shape[0]).reshape(x.shape[0], 1)), axis=1)

    # minibatch gradient descent algorithm
    if len(sys.argv) == 1:
        batch_size = xdata.shape[0]//10         # 10 batches
    else:
        batch_size = int(sys.argv[1])

    print(batch_size)

    eta = 0.1 # learning rate
    n_iterations = 500      # this many runs
    N = xdata.shape[0]      # data length

    # initialize weights
    w = np.random.randn(xdata.shape[1])

    # find weights iteratively
    for i in range(n_iterations):
        mini_batches = create_mini_batches(xdata, z.reshape(z.shape[0], 1), batch_size)

        for mini_batch in mini_batches:
            x_mini, z_mini = mini_batch
            error = np.matmul(x_mini, w) - z_mini
            gradient = 2.0 * np.matmul(x_mini.transpose(),error) / batch_size
            w = w - eta * gradient

    print(w)

    print("Calculated Weight : ", w)

    accuracy = calculate_accuracy()
    print("Accuracy of the model ( Minibatch Gradient Descent ) : ", accuracy)


except:
    ex = sys.exc_info()
    print("EXception Raised !!! ")
    print("Exception Details  :  ", ex[0])