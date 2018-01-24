import tensorflow as tf
import numpy as np

# Define parameters

batch_size = 128
test_size = 256
img_size = 28
num_classes = 10
logs_path= 'mnist_data_graph'

# Initialize weight/filter/kernel for convolving

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Network model for digit recognition Conv1 -> Relu1 -> Max_pool1 -> dropout1 ->  Conv2 -> Relu2 -> Max_pool2 -> dropout2 ->  Conv3 -> Relu3 -> Max_pool3 -> dropout3 ->FC->FC

def model(X, w1, w2, w3, w4, wf, prob_keep, prob_hidden):

    conv1 = tf.nn.conv2d(X, w1,\
                         strides=[1, 1, 1, 1],\
                         padding='SAME')

    conv1_activation = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_activation, ksize=[1, 2, 2, 1]\
                        ,strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv1 = tf.nn.dropout(conv1, prob_keep)

    conv2 = tf.nn.conv2d(conv1, w2,\
                         strides=[1, 1, 1, 1],\
                         padding='SAME')
    conv2_activation = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_activation, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv2 = tf.nn.dropout(conv2, prob_keep)

    conv3=tf.nn.conv2d(conv2, w3,\
                       strides=[1, 1, 1, 1]\
                       ,padding='SAME')

    conv3 = tf.nn.relu(conv3)


    Fully_connected_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')

    Fully_connected_layer = tf.reshape(Fully_connected_layer, [-1, w4.get_shape().as_list()[0]])
    Fully_connected_layer = tf.nn.dropout(Fully_connected_layer, prob_keep)


    output_layer = tf.nn.relu(tf.matmul(Fully_connected_layer, w4))
    output_layer = tf.nn.dropout(output_layer, prob_hidden)

    result = tf.matmul(output_layer, wf)
    return result

# Import data set

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

train_x, train_y, test_x, test_y = mnist.train.images,\
                     mnist.train.labels, \
                     mnist.test.images, \
                     mnist.test.labels
# Reshape dataset

train_x = train_x.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img
test_x = test_x.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img

# Create placeholder and shape of weights

X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])

weight_l1 = init_weights([5, 5, 1, 32])       # 5x5x1 conv, 32 outputs
weight_l2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
weight_l3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
weight_l4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
weight_final = init_weights([625, num_classes])         # FC 625 inputs, 10 outputs (labels from 0 to 9)

# Dropout to reduce overfitting. It randomly deads some neurons in the network.

prob_keep = tf.placeholder("float")
prob_hidden = tf.placeholder("float")
model_train = model(X, weight_l1, weight_l2, weight_l3, weight_l4, weight_final, prob_keep, prob_hidden)

# Calculate the loss and un optimization algorithm

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=model_train, labels=Y)
cost = tf.reduce_mean(Y_)
optimizer  = tf.train.\
           RMSPropOptimizer(0.001, 0.9).minimize(cost)
tf.summary.scalar("cost", Y_)
tf.summary.scalar("accuracy", cost)
predict_op = tf.argmax(model_train, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(logs_path, \
                                    graph=tf.get_default_graph())
    for i in range(10):
        training_batch = \
                       zip(range(0, len(train_x), \
                                 batch_size),
                             range(batch_size, \
                                   len(train_x)+1, \
                                   batch_size))
        for start, end in training_batch:
            sess.run(optimizer, feed_dict={X: train_x[start:end],\
                                          Y: train_y[start:end],\
                                          prob_keep: 0.8,\
                                          prob_hidden: 0.2})

        test_indices = np.arange(len(test_x))# Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(test_y[test_indices], axis=1) ==\
                         sess.run\
                         (predict_op,\
                          feed_dict={X: test_x[test_indices],\
                                     Y: test_y[test_indices], \
                                     prob_keep: 1.0,\
                                     prob_hidden: 1.0})))
