import numpy as np
import tensorflow as tf
import data_augmentor
import data
import plot_utils
import print_utils
import model_config
from time import time

np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)
tf.set_random_seed(42)
MODEL_NAME = "cnn_add_data"

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
    return tf.nn.max_pool(X, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, stride, stride, 1], padding=padding)
    
def main():
    # reset tf graph
    tf.reset_default_graph()

    # get model configuration
    model_configs = model_config.cnn_baseline

    # load data
    train, valid, test =\
    data.load_data_with_augmentation(n_train_samples_per_class=model_configs['n_train_samples_per_class'],
                  classes=np.asarray(model_configs['classes']))

    train_shift = data_augmentor.data_preprocessing_augmentor(train, random_shift=True, random_rotation=False,\
                                random_shear=False, random_zoom=False)
    train_rotate = data_augmentor.data_preprocessing_augmentor(train, random_shift=False, random_rotation=True,\
                                random_shear=False, random_zoom=False)
    train_shear = data_augmentor.data_preprocessing_augmentor(train, random_shift=False, random_rotation=False,\
                                random_shear=True, random_zoom=False)
    train_zoom = data_augmentor.data_preprocessing_augmentor(train, random_shift=False, random_rotation=False,\
                                random_shear=False, random_zoom=True)

    print("original set")
    plot_utils.plot_images_side_by_side(train.images[0:10,:], prefix="original_train")
    plot_utils.plot_images_side_by_side(valid.images[0:10,:], prefix="original_valid")
    plot_utils.plot_images_side_by_side(test.images[0:10,:], prefix="original_test")
    print("train_shift")
    plot_utils.plot_images_side_by_side(train_shift.images[0:10,:], prefix="train_shift")
    print("train_rotate")
    plot_utils.plot_images_side_by_side(train_rotate.images[0:10,:], prefix="train_rotate")
    print("train_shear")
    plot_utils.plot_images_side_by_side(train_shear.images[0:10,:], prefix="train_shear")
    print("train_zoom")
    plot_utils.plot_images_side_by_side(train_zoom.images[0:10,:], prefix="train_zoom")

    # get number of samples per dataset
    n_train_samples = train.images.shape[0]
    n_valid_samples = valid.images.shape[0]
    n_test_samples = test.images.shape[0]
    
    # define input and output
    input_width = model_configs['input_width']
    n_input = model_configs['n_input']
    n_classes = np.asarray(model_configs['classes']).shape[0]
    
    # define training hyper-parameters
    n_epochs = model_configs['n_epochs']
    minibatch_size = model_configs['minibatch_size']
    learning_rate = model_configs['learning_rate']
    regularization_term = model_configs['regularization_term']
    keep_probability = model_configs['keep_probability']
    
    #define conv layer architecture
    filter_size = model_configs['filter_size']
    num_filters = model_configs['num_filters']
    conv_stride = model_configs['conv_stride']
    max_pool_stride = model_configs['max_pool_stride']
    pool_size = model_configs['pool_size']
    padding = model_configs['padding']
    
    # define FC NN architecture
    fc1_size = model_configs['fc1_size']
    fc2_size = model_configs['fc2_size']

    # define visualziation parameters
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
        W_fc1 = weight_variable([max_pool_out_dim*max_pool_out_dim*num_filters, fc1_size])
        b_fc1 = bias_variable([fc1_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # fully connected layer 2
    with tf.variable_scope("fc_2"):
        W_fc2 = weight_variable([fc1_size, fc2_size])
        b_fc2 = bias_variable([fc2_size])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
        h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob=keep_prob)

    # output layer
    with tf.variable_scope("output_1"):
        W_fc3 = weight_variable([fc2_size, n_classes])
        b_fc3 = bias_variable([n_classes])
        y_conv = tf.matmul(h_fc2_dropout, W_fc3) + b_fc3
    
    # compute losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
    W1 = tf.get_default_graph().get_tensor_by_name("fc_1/weight:0")
    W2 = tf.get_default_graph().get_tensor_by_name("fc_2/weight:0")
    W3 = tf.get_default_graph().get_tensor_by_name("output_1/weight:0")
    reg_loss = tf.reduce_sum(tf.pow(tf.abs(W1),2)) + tf.reduce_sum(tf.pow(tf.abs(W2),2)) + \
                            tf.reduce_sum(tf.pow(tf.abs(W3),2))
    cost = cross_entropy + (reg_loss * regularization_term)
    
    # compute predictions and error
    prediction = tf.argmax(y_conv, axis=1)
    correct = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y, axis=1))
    error = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))

    # compute confusion matrix
    confusion_matrix = tf.confusion_matrix(tf.argmax(y, axis=1), prediction, num_classes=n_classes)
        
    # training op
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # initialize variables and session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init.run()
        
        # initialize cost and error variables
        train_iteration_errors = []
        train_errors = []
        valid_errors = []
        test_errors = []
        
        # calculate number of iterations per epoch
        train_iterations = int(n_train_samples / minibatch_size)
        
        start_time = time()
        for epoch in range(n_epochs):
            if(epoch % 10 == 0):
                print("--- epoch: {}".format(epoch))
            # reset error each epoch
            epoch_train_error = 0.
            epoch_valid_error = 0.
            epoch_test_error = 0.
        
            for i in range(train_iterations):
                # Get next batch of original training data and labels
                train_data_mb, train_label_mb = train.next_batch(minibatch_size)
                # compute error
                train_mb_error = error.eval(feed_dict={X: train_data_mb, y: train_label_mb, keep_prob: 1.0})
                epoch_train_error += train_mb_error
                train_iteration_errors.append(train_mb_error)

                # training operation - original data
                sess.run(optimizer, feed_dict={X: train_data_mb, y: train_label_mb, keep_prob: keep_probability})
                
                # Get next batch of gaussian noise training data and labels
                train_shift_data_mb, train_shift_label_mb = train_shift.next_batch(minibatch_size)
                # training operation - guassian noise data
                sess.run(optimizer, feed_dict={X: train_shift_data_mb, y: train_shift_label_mb, keep_prob: keep_probability})
                
                # Get next batch of poisson noise training data and labels
                train_rotate_data_mb, train_rotate_label_mb = train_rotate.next_batch(minibatch_size)
                # training operation - poisson noise data
                sess.run(optimizer, feed_dict={X: train_rotate_data_mb, y: train_rotate_label_mb, keep_prob: keep_probability})
                
                # Get next batch of flip left-right training data and labels
                train_shear_data_mb, train_shear_label_mb = train_shear.next_batch(minibatch_size)
                # training operation - flip left-right noise data
                sess.run(optimizer, feed_dict={X: train_shear_data_mb, y: train_shear_label_mb, keep_prob: keep_probability})
                
                # Get next batch of swirl-rotate training data and labels
                train_zoom_data_mb, train_zoom_label_mb = train_zoom.next_batch(minibatch_size)
                # training operation - swirl-rotate noise data
                sess.run(optimizer, feed_dict={X: train_zoom_data_mb, y: train_zoom_label_mb, keep_prob: keep_probability})
        
            # compute average train epoch error
            train_errors.append(epoch_train_error / train_iterations)
            
            # compute valid epoch error through mini-batches
            valid_iterations = int(n_valid_samples / minibatch_size)
            for i in range (valid_iterations):
                valid_data_mb, valid_label_mb = valid.next_batch(minibatch_size)
                valid_mb_error = error.eval(feed_dict={X: valid_data_mb, y: valid_label_mb, keep_prob: 1.0})
                epoch_valid_error += valid_mb_error
            avg_epoch_valid_error = epoch_valid_error / valid_iterations
            valid_errors.append(avg_epoch_valid_error)

            # compute test epoch error through mini-batches
            test_iterations = int(n_test_samples / minibatch_size)
            for i in range (test_iterations):
                test_data_mb, test_label_mb = test.next_batch(minibatch_size)
                test_mb_error = error.eval(feed_dict={X: test_data_mb, y: test_label_mb, keep_prob: 1.0})
                epoch_test_error += test_mb_error
            avg_epoch_test_error = epoch_test_error / test_iterations
            test_errors.append(avg_epoch_test_error)
        end_time = time()

        # print training time
        print("training time: {0:.2f} secs".format(end_time-start_time))
        
        # save final model
        save_path = saver.save(sess, "./models/{}_final.ckpt".format(MODEL_NAME))

        # plot confusion matrix
        confusion_mat = confusion_matrix.eval(feed_dict={X: test.images, y: test.labels, keep_prob: 1.0})
        plot_utils.plot_confusion_matrix(confusion_mat)
        
        # print final errors
        print_utils.print_final_error(train_errors[-1], valid_errors[-1], test_errors[-1])
        # print test error based on best valid epoch
        print_utils.print_best_valid_epoch(train_errors, valid_errors, test_errors)
        print_utils.write_errors_to_file(train_errors, valid_errors, test_errors, model_configs, MODEL_NAME)
        
        # plot error vs. epoch
        plot_utils.plot_epoch_errors(train_errors, valid_errors, prefix=MODEL_NAME)
        plot_utils.plot_train_iteration_errors(train_iteration_errors, prefix=MODEL_NAME)
        plot_utils.plot_cnn_kernels(vis_layers, W_conv1, prefix=MODEL_NAME)

if __name__ == "__main__":
    main()