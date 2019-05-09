import numpy as np

# set up input here
train_data = np.load("concaveData.npy")
train_label = np.load("concaveTarget.npy")
N, p = train_data.shape

# set up test data
test_date = np.load("TestData.npy")
test_label = np.load("TestTarget.npy")


const = 509.0


def modify_data_and_save(file_name, data_set):

    if data_set.shape.__len__()>1:
        for i in range(data_set.shape[0]):
            for k in range(data_set.shape[1]):
                data_set[i, k] = data_set[i, k]+const

    np.save('modified_'+file_name, data_set)


modify_data_and_save("concaveData.npy", train_data)
modify_data_and_save("concaveTarget.npy", train_label)
modify_data_and_save("TestData.npy", test_date)
modify_data_and_save("TestTarget.npy", test_label)