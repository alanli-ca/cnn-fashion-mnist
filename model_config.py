cnn_baseline = {
    # define input and output
    'input_width': 28,
    'n_input': 784,
    'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # [1, 4, 5, 8, 9] reduced set w/o ambiguity
    'n_train_samples_per_class': 5000,
    
    # define training hyper-parameters
    'n_epochs': 25,
    'minibatch_size': 500, # default 500
    'learning_rate': 0.001, # default 0.002
    'regularization_term': 0.001, # default 0.001
    'keep_probability': 1.0, # default 0.5
    
    #define conv layer architecture
    'filter_size': 5, # default 5
    'num_filters': 32, # default 32
    'conv_stride': 1, # default 1
    'max_pool_stride': 2, # default 2
    'pool_size': 3, # default 3
    'padding': "VALID",
    
    # define FC NN architecture
    'fc1_size': 384,
    'fc2_size': 192
}

cnn_2x_scale = {
    # define input and output
    'input_width': 56,
    'n_input': 3136,
    'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # [1, 4, 5, 8, 9] reduced set w/o ambiguity
    'n_train_samples_per_class': 1000,
    
    # define training hyper-parameters
    'n_epochs': 100,
    'minibatch_size': 500,
    'learning_rate': 0.001,
    'regularization_term': 0.01,
    'keep_probability': 1.0,
    
    #define conv layer architecture
    'filter_size': 5, # default 5
    'num_filters': 32, # default 32
    'conv_stride': 1, # default 1
    'max_pool_stride': 2, # default 2
    'pool_size': 3, # default 3
    'padding': "VALID",
    
    # define FC NN architecture
    'fc1_size': 784,
    'fc2_size': 196
}