import numpy as np
import sys
import tensorflow as tf

# set up input here
train_data = np.load("concaveData.npy")
train_label = np.load("concaveTarget.npy")
N, p = train_data.shape

# set up test data
test_data = np.load("TestData.npy")
test_label = np.load("TestTarget.npy")


# minibatch gradient descent algorithm
if len(sys.argv) == 1:
    batch_size = train_data.shape[0] // 10 # 10 batches
else:
    batch_size = int(sys.argv[1])
    print(batch_size)


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


# set up network
n_inputs = p
n_hidden1 = 250
n_hidden2 = 125
n_outputs = 3   # 3 classes

# use placeholder nodes to represent training data and targets
# shape of input is (None, n_inputs)
# assume training data is scaled to [0,1] floating point
# during execution phase, X will be replaced with one training batch at a time
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
t = tf.placeholder(tf.int64, shape= None, name="t")

# create two hidden layers and output layer
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

# define cost function to train network
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# define how to train
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = optimizer.minimize(loss)



# how to evaluate model
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, t, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# initialize
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 500
is_converge = 0
error_threshold = 0.05
convergence_epoch = -1
max_accuracy = -1.0

try:
    with tf.Session() as session:
        init.run()
        for epoch in range(n_epochs):
            mini_batches = create_mini_batches(train_data, train_label.reshape(train_label.shape[0], 1), batch_size)
            for mini_batch in mini_batches:
                # do training step first
                X_batch, t_batch = mini_batch
                _, loss_curr = session.run([training_step, loss], feed_dict={X:X_batch, t:t_batch})

                # check accuracies training and test
                acc_train = accuracy.eval(feed_dict={X:X_batch, t:t_batch})
                acc_val = accuracy.eval(feed_dict={X:test_data, t:test_label})
                max_accuracy = max(max_accuracy, acc_val)

                # to determine the convergence point
                if is_converge == 0 and loss_curr < error_threshold:
                    is_converge = 1
                    convergence_epoch = epoch

            print('epoch :', epoch, '   Loss : ', "{0:.3f}".format(round(loss_curr, 3)),
                '   Accuracy for Train Data : ', "{0:.3f}".format(round(acc_train, 3)),
                  '    Accuracy for Test Data : ', "{0:.3f}".format(round(acc_val, 3)))

        if is_converge == 1:
            print("The Model converged at epoch : ", convergence_epoch)
        else:
            print("The Model doesn't converge !")

        print('Maximum accuracy of the DNN Model (2 Hidden Layer with default config): ', max_accuracy)

        save_path = saver.save(session,"./model_final.ckpt")

except:
    ex = sys.exc_info()
    print("Exception Details  :  ", ex[0])