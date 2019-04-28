import numpy as np
import scipy
import sklearn


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

    # compute pseudo-inverse
    xtx = np.matmul(xdata.transpose(), xdata)
    xtxinv = np.linalg.inv(xtx)
    xinv = np.matmul(xtxinv, xdata.transpose())

    # compute least squares solution
    w = np.matmul(xinv, z)
    print("Wight : ", w)

    accuracy = calculate_accuracy()
    print("")
    print("Accuracy of the model (Pseudo Inverse Linear Classifier) : ", accuracy)

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])