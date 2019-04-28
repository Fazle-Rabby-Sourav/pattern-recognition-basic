# generate samples of two classes
#   a. Gaussian densities
#   b. class means [1.0, -1.0] and [-1.0, 1.0]
#   c. variances [1.0, 1.0] and [1.3, 1.3]

# use 10,000 samples of 5,000 Class-0 and 5,000 Class-1 to train a Perceptron to classify the samples;
# report the separating line parameters;
# report the accuracy report the average accuracies,
# report the precision and recall scores

import math
import numpy as np
import scipy
import sklearn

import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


def plot_data_set(data_mat1, data_mat2):
    x, y = data_mat1.T
    plt.scatter(x, y, c='red')

    x, y = data_mat2.T
    plt.scatter(x, y, c='blue')

    plt.xlim([-6.0, 6.0])
    plt.ylim([-6.0, 6.0])

num_data_first_class = 5000
num_data_second_class = 5000

num_test_data = 500

number_of_data = num_data_first_class + num_data_second_class

# Calculating Standard Variation From Variance
sd_x_first = math.sqrt(1);
sd_y_first = math.sqrt(1);
sd_x_second = math.sqrt(1.3);
sd_y_second = math.sqrt(1.3);

#   Generate class 0 Train Data : Normal distribution
x0data = np.transpose(np.array([[np.random.normal(1.0, sd_x_first) for i in range(num_data_first_class)],
                                [np.random.normal(-1.0, sd_y_first) for i in range(num_data_first_class)]]))
t0vec = -np.ones(num_data_first_class)

#   Generate class 0 Test Data
test0Data = np.transpose(np.array([[np.random.normal(1.0, sd_x_first) for i in range(num_test_data)],
                                [np.random.normal(-1.0, sd_y_first) for i in range(num_test_data)]]))
test0Label = -np.ones(num_test_data)


#   generate class 1 Train Data :  Normal distribution
x1data = np.transpose(np.array([[np.random.normal(-1.0, sd_x_second) for i in range(num_data_second_class)],
                                [np.random.normal(1.0, sd_y_second) for i in range(num_data_second_class)]]))
t1vec = np.ones(num_data_second_class)

#   generate class 1 Test Data
test1Data = np.transpose(np.array([[np.random.normal(-1.0, sd_x_second) for i in range(num_test_data)],
                                [np.random.normal(1.0, sd_y_second) for i in range(num_test_data)]]))
test1Label = np.ones(num_test_data)


plot_data_set(x0data, x1data)
plt.savefig('Points for perceptro.png')

#   Train Data for Both class
xdata = np.concatenate((x1data, x0data), axis=0)
tvec = np.concatenate((t1vec, t0vec))

#   Test Data fot both class
testData = np.concatenate((test1Data, test0Data), axis=0)
testLabel = np.concatenate((test1Label, test0Label), axis=0)

print("xdata's shape : ", xdata.shape, "xdata : ", xdata)
print("tvec's shape : ", tvec.shape, "tvec : ", tvec)

shuffle_index = np.random.permutation(number_of_data)
xdata, tvec = xdata[shuffle_index], tvec[shuffle_index]

#   Train the classifier
classifier = Perceptron(tol=1e-3, random_state=0)

print("\n\n******************************************* Paramesters ****************************************************")
print("Perceptron Parameters    : ", classifier.fit(xdata, tvec))
print("Classiifer Coefficient   : ", classifier.coef_)
print("Classiifer Intercept     : ", classifier.intercept_)
print("Classiifer Iteration     : ", classifier.n_iter_)
print("Classiifer Correctness   : ",  np.equal(classifier.predict(xdata), tvec))


print("\n\n********************************************* Accuracy *****************************************************")
#   Accuracy Score: mean accuracy on the given test data and labels
print("Accuracy                                                     : ", classifier.score(testData, testLabel))

#   n-fold cross validation : Accuracy
print("Average of accuracies found from 10-fold cross-validation    : ",
      np.average(cross_val_score(classifier, xdata, tvec, cv=10, scoring="accuracy")))


print("\n\n**************************************** Precision and recall *********************************************")
#   predicted value instead of Accuracy
predicted_values = cross_val_predict(classifier, xdata, tvec, cv=10)
print("Predicted Value                  : ", predicted_values)

#   Confusion Matrix
print("Corresponding Confusion Matrix : ",  confusion_matrix(tvec, predicted_values));

#   precision and recall
print("\n")
print("Precision Score                : ", precision_score(tvec, predicted_values))
print("Recall Score                   : ", recall_score(tvec, predicted_values))