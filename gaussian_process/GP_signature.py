import sys

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)



def read_file_and_convert_to_list(fname):

    with open(fname) as f:
        content = f.readlines()

    temp = content[0].split('], [')
    X_list = list()
    Y_list = list()

    for item in temp:
        item = item.replace(",", "")
        item = item.replace("[", "")
        item = item.replace("]", "")
        str_list = item.split(" ")
        x = int(str_list[0])
        y = int(str_list[1])

        X_list.append(x)
        Y_list.append(y)
        print(item)

    return X_list, Y_list


try:
    X, y = read_file_and_convert_to_list("37")

    X = np.atleast_2d(X)
    y = np.atleast_2d(y)

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    plt.figure()
    # plt.plot(X, y, 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=1, label='Observations')
    plt.ylim(0, 149)
    print("Completed")

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])