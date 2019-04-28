import math
import numpy as np
from sklearn.linear_model import Perceptron
import sys

try:
    # Input
    xdata = np.load('train_data.npy')

    # Observed Label
    label = np.load('train_data_label.npy')

    #   Train the Perceptron
    classifier = Perceptron(tol=1e-3, random_state=0)

    print("\n\n******************************************* Paramesters ****************************************************")
    print("Perceptron Parameters    : ", classifier.fit(xdata, label))

    testData = np.load('test_data.npy')
    testLabel = np.load('test_data_label.npy')
    print("\n\n********************************************* Accuracy *****************************************************")
    #   Accuracy Score: mean accuracy on the given test data and labels
    print("Accuracy : ", classifier.score(testData, testLabel))

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])