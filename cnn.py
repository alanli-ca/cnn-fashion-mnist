import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mnist import MNIST
from tensorflow.examples.tutorials.mnist import input_data
import data_augmentor

np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)
tf.set_random_seed(42)

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer(uniform=False, seed=42)
    W = tf.get_variable("weight", shape=shape, initializer=initial)
    return W

def bias_variable(shape):
    initial = tf.zeros_initializer()
    b = tf.get_variable("bias", shape=shape, initializer=initial)
    return b

def conv2d(X, W, stride=1, padding="VALID"):
    return tf.nn.conv2d(X, W, strides=[1,stride,stride,1], padding=padding)

def max_pool(X, stride=2, pool_size=3, padding="VALID"):
    return tf.nn.max_pool(X, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding=padding)

def print_cost_and_error(train_costs, train_errors, test_costs, test_errors):
    print("final_traing_cost: {0:0.3f} | final_traing_error: {1:0.3f}".format(train_costs[-1], train_errors[-1]))
    print("final_test_cost: {0:0.3f} | final_test_error: {1:0.3f}".format(test_costs[-1], test_errors[-1]))

def plot_figures(train_costs, train_errors):
    
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_costs, label="train")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost vs. Epoch")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(train_errors, label="train")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error")
    ax2.set_title("Error vs. Epoch")
    ax2.set_ylim([0, 1])
    ax2.legend()

    plt.tight_layout()
    plt.savefig("{}.png".format("cnn"))
    plt.show()

def main():
    # reset tf graph
    tf.reset_default_graph()

    # load data TODO
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)
    train = fashion_mnist.train
    test = fashion_mnist.test
    trainData = fashion_mnist.train.images
    trainTarget = fashion_mnist.train.labels
    testData = fashion_mnist.test.images
    testTarget = fashion_mnist.test.labels
    
    train_idx = np.arange(trainData.shape[0]) 
    
    # define input and output
    input_width = 28
    n_input = input_width*input_width
    n_classes = 10
    
    (n_x, m) = train.images.T.shape
    
    # define hyper-parameters
    minibatch_size = 500
    filter_size = 5 # default 5
    num_filters = 32 # default 32
    conv_stride = 1 # default 1
    max_pool_stride = 2 # default 2
    pool_size = 3 # default 3
    padding = "VALID"
    learning_rate = 0.002
    regularization_term = 0.001
    n_epochs = 1
    keep_probability = 0.5
    vis_layers = np.arange(0,8) # selected filter visualization layers
    
    # define placeholders
    X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
    y = tf.placeholder(tf.int32, shape=(None, n_classes), name="y")
    keep_prob = tf.placeholder(tf.float32)
    
    # input reshaping
    X_image = tf.reshape(X, [-1, input_width, input_width, 1])
    
    # convolutional layer 1
    with tf.variable_scope("conv_1"):
        W_conv1 = weight_variable([filter_size, filter_size, 1, num_filters])
        b_conv1 = bias_variable([num_filters])
        h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1, stride=conv_stride, padding=padding) + b_conv1)
        
    # convolutional output dimension
    conv_out_dim = np.int(np.floor((input_width-filter_size)/conv_stride + 1))
    # max pooling output dimension
    max_pool_out_dim = np.int(np.floor((conv_out_dim-pool_size)/max_pool_stride + 1))
    
    # max pooling layer 1
    with tf.variable_scope("pool_1"):
        h_pool1 = max_pool(h_conv1, stride=max_pool_stride, pool_size=pool_size, padding=padding)    
        h_pool1_flat = tf.reshape(h_pool1, [-1, max_pool_out_dim*max_pool_out_dim*num_filters])
    
    # fully connected layer 1
    with tf.variable_scope("fc_1"):
        W_fc1 = weight_variable([max_pool_out_dim*max_pool_out_dim*num_filters, 384])
        b_fc1 = bias_variable([384])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # fully connected layer 2
    with tf.variable_scope("fc_2"):
        W_fc2 = weight_variable([384, 192])
        b_fc2 = bias_variable([192])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
        h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob=keep_prob)

    # output layer
    with tf.variable_scope("output_1"):
        W_fc3 = weight_variable([192, n_classes])
        b_fc3 = bias_variable([n_classes])
        y_conv = tf.matmul(h_fc2_dropout, W_fc3) + b_fc3
    
    # compute losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    W1 = tf.get_default_graph().get_tensor_by_name("fc_1/weight:0")
    W2 = tf.get_default_graph().get_tensor_by_name("fc_2/weight:0")
    W3 = tf.get_default_graph().get_tensor_by_name("output_1/weight:0")
    reg_loss = tf.reduce_sum(tf.pow(tf.abs(W1),2)) + tf.reduce_sum(tf.pow(tf.abs(W2),2)) + tf.reduce_sum(tf.pow(tf.abs(W3),2))
    cost = cross_entropy + (reg_loss * regularization_term)
    
    # compute error
    correct = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y, axis=1))
    error = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
        
    # training op
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # initialize variables and session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        
        # initialize cost and error variables
        train_costs = []
        train_errors = []
        test_costs = []
        test_errors = []
        
        for epoch in range(n_epochs):
            print("--- epoch: {}".format(epoch))
            epoch_train_cost = 0.
            epoch_train_error = 0.
            num_minibatches = int(m / minibatch_size)
        
            for i in range(num_minibatches):
                # Get next batch of training data and labels        
                X_mb, y_mb = train.next_batch(minibatch_size)
                mb_train_cost = cost.eval(feed_dict={X: X_mb, y: y_mb, keep_prob: 1.0})
                mb_train_error = error.eval(feed_dict={X: X_mb, y: y_mb, keep_prob: 1.0})

                epoch_train_cost += mb_train_cost
                epoch_train_error += mb_train_error

                # training
                sess.run(optimizer, feed_dict={X: X_mb, y: y_mb, keep_prob: keep_probability})

            train_costs = np.append(train_costs, epoch_train_cost)
            train_errors = np.append(train_errors, epoch_train_error / num_minibatches)

        # save the model
        # save_path = saver.save(sess, "./models/cnn_final.ckpt")
        
        test_costs = np.append(test_costs, cost.eval(feed_dict={X: testData, y: testTarget, keep_prob: 1.0}))
        test_errors = np.append(test_errors, error.eval(feed_dict={X: testData, y: testTarget, keep_prob: 1.0}))

        print_cost_and_error(train_costs, train_errors, test_costs, test_errors)
        
        # plot loss and error vs. epoch
        plot_figures(train_costs, train_errors)
        
        # plot cnn kernels
        fig = plt.figure(figsize=(8, 1))
        for i in range(vis_layers.shape[0]):
            ax = fig.add_subplot(1, 8, i+1)
            ax.imshow(tf.squeeze(W_conv1)[:,:,vis_layers[i]].eval(), cmap='gray')
            ax.axis("off")
        plt.savefig("filter_viz.png")    
        plt.show()
if __name__ == "__main__":
    main()