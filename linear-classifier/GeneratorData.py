import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_data_first_class = 200
num_data_second_class = 100
number_of_data = num_data_first_class + num_data_second_class


def plot_data_set(data_mat1, data_mat2):
    x, y = data_mat1.T
    plt.scatter(x, y, c='red')

    x, y = data_mat2.T
    plt.scatter(x, y, c='blue')

    plt.xlim([-6.0, 6.0])
    plt.ylim([-6.0, 6.0])

# Save datasets as npy file
def save_datasets_as_file():
    np.save('train_data.npy', x_data_train)
    np.save('train_data_label.npy', x_label_train)
    np.save('test_data.npy', x_data_test)
    np.save('test_data_label.npy', x_label_test)


# -------------------------------- Class 0 --------------------------------- #
# class 0 : Uniform distribution
range_low = np.array([-3.0, -3.0])
range_high = np.array([-1.0, 3.0])
sz = np.array([100, 2])
x0_first_part = np.random.uniform(range_low, range_high, sz)

range_low = np.array([-1.0, -3.0])
range_high = np.array([3.0, -1.0])
sz = np.array([100, 2])
x0_second_part = np.random.uniform(range_low, range_high,sz)

x0_data = np.concatenate((x0_first_part, x0_second_part), axis=0)
x0_label = -np.ones(num_data_first_class)


# -------------------------------- Class 1 --------------------------------- #
# class 1 : Normal distribution
means = np.array([1.5, 1.5])
sd = np.array([0.6, 0.6])
sz = np.array([100, 2])

x1_data = np.random.normal(means, sd, sz)
x1_label = np.ones(num_data_second_class)


# Ploting the points
plot_data_set(x0_data, x1_data)
plt.savefig('data_distribution_figure.png')


# Merge data from two class
x_data = np.concatenate((x1_data, x0_data), axis=0)
x_label = np.concatenate((x1_label, x0_label))


# Shuffling the data
shuffle_index = np.random.permutation(number_of_data)
x_data, x_label = x_data[shuffle_index], x_label[shuffle_index]


# Split data and label into train and test
x_data_train, x_data_test, x_label_train, x_label_test = train_test_split(x_data, x_label, test_size=0.30, random_state=42)

# Save as npy file
save_datasets_as_file()
