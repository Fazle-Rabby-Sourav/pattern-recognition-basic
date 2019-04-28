import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
import sys
from sklearn import svm, linear_model, metrics


def activation_funtion(val):
    if val >= 0:
        return 1
    else:
        return -1


def plot_decision_boundary(clf, name_of_clf, X, Y, cmap='Paired_r'):
    h = 0.15
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.4)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')

    plt.savefig(name_of_clf)

def calculate_accuracy(classifier):

    testData = np.load('test_data.npy')
    testLabel = np.load('test_data_label.npy')

    cnt_right_prediction = 0
    cnt_wrong_prediction = 0

    for idx, point in enumerate(testData):
        predicted_label = int(classifier.predict([point]))
        actual_label = int(testLabel[idx])

        if actual_label == predicted_label:
            # print(point, "Right Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label)
            cnt_right_prediction = cnt_right_prediction + 1
        else:
            # print(point, "Wrong Prediction !! Actual Label : ", actual_label, "Predicted Label : ", predicted_label)
            cnt_wrong_prediction = cnt_wrong_prediction + 1

    accuracy = cnt_right_prediction/( cnt_right_prediction+cnt_wrong_prediction )
    return accuracy


try:
    # Input
    xdata = np.load('train_data.npy')
    label = np.load('train_data_label.npy')

    # Test Data
    testData = np.load('test_data.npy')
    testLabel = np.load('test_data_label.npy')

    accuracy_of_classfier = []
    classifier_name = ["Perceptron-1", "Perceptron-2", "SGD Linear Classifier", "Linear SVM Classifer"]

    # ****************************** Perceptron-1****************************** #
    perceptron_classifier1 = Perceptron(tol=1e-3, eta0=0.001, random_state=None, fit_intercept=1, max_iter=500000)
    perceptron_classifier1.fit(xdata, label)
    acc = perceptron_classifier1.score(testData, testLabel)
    accuracy_of_classfier.append(acc)

    print("Accuracy of Perceptron-1 Classifier : ", accuracy_of_classfier[0] )
    plot_decision_boundary(perceptron_classifier1, classifier_name[0], testData, testLabel)

    # ****************************** Perceptron-2 ****************************** #
    perceptron_classifier2 = Perceptron(tol=1e-3, eta0=0.0005, random_state=None, fit_intercept=1, max_iter=500000)
    perceptron_classifier2.fit(xdata, label)
    acc = perceptron_classifier2.score(testData, testLabel)
    accuracy_of_classfier.append(acc)
    print("Accuracy of Perceptron-2 Classifier : ", perceptron_classifier2.score(testData, testLabel))
    plot_decision_boundary(perceptron_classifier2, classifier_name[1], testData, testLabel)

    # ****************************** SGD Linear Classifier ******************************
    SGD_classifier = linear_model.SGDClassifier(tol=1e-3, max_iter=500000)
    SGD_classifier.fit(xdata, label)
    SGD_prediction = SGD_classifier.predict(testData)
    acc = metrics.accuracy_score(testLabel, SGD_prediction)
    accuracy_of_classfier.append(acc)
    print("Accuracy of SGD Classifier : ", accuracy_of_classfier[2])
    plot_decision_boundary(SGD_classifier, classifier_name[2], testData, testLabel)

    # ****************************** SVM ****************************** #
    svm_classifier = LinearSVC(random_state=0, tol=1e-5)
    svm_classifier.fit(xdata, label)
    acc = calculate_accuracy(svm_classifier)
    accuracy_of_classfier.append(acc)
    print("Accuracy of SVM Classifier : ", accuracy_of_classfier[3])
    plot_decision_boundary(svm_classifier, classifier_name[3], testData, testLabel)

    best_classifier = ""
    best_accuracy = 0.0
    for i, acc in enumerate(accuracy_of_classfier):
        if acc > best_accuracy:
            best_accuracy = max(acc, best_accuracy)
            best_classifier = classifier_name[i]

    print("Best Classifier : ", best_classifier)
    print("Accuracy : ", best_accuracy)

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])