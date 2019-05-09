# generate samples of two classes
#   a. Gaussian densities
#   b. class means [1.0, -1.0] and [-1.0, 1.0]
#   c. variances [1.0, 1.0] and [1.3, 1.3]

# use 10,000 samples of 9,000 Class-0 and 1,000 Class-1 to train a Perceptron to classify the samples;
# report the separating line parameters;
# report the accuracy report the average accuracies,
# report the precision and recall scores


import numpy as np
import scipy
import  sklearn
import math

from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

num_data_first_class = 5000
num_data_second_class = 5000

number_of_data = num_data_first_class + num_data_second_class

# Calculating Standard Variation From Variance
sd_x_first = math.sqrt(1);
sd_y_first = math.sqrt(1);
sd_x_second = math.sqrt(1.3);
sd_y_second = math.sqrt(1.3);

#   Generate class 0 Data : Normal distribution
x0data = np.transpose(np.array([[np.random.normal(1.0, sd_x_first) for i in xrange(num_data_first_class)],
                                [np.random.normal(-1.0, sd_y_first) for i in xrange(num_data_first_class)]]))
t0vec = -np.ones(num_data_first_class)


#   generate class 1 Data :  Normal distribution
x1data = np.transpose(np.array([[np.random.normal(-1.0, sd_x_second) for i in xrange(num_data_second_class)],
                                [np.random.normal(1.0, sd_y_second) for i in xrange(num_data_second_class)]]))
t1vec = np.ones(num_data_second_class)

#   Both class
xdata = np.concatenate((x1data, x0data), axis=0)
tvec = np.concatenate((t1vec, t0vec))

print("xdata's shape : ", xdata.shape, "xdata : ", xdata)
print("tvec's shape : ", tvec.shape, "tvec : ", tvec)

shuffle_index = np.random.permutation(number_of_data)
xdata, tvec = xdata[shuffle_index], tvec[shuffle_index]