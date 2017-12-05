cnn_baseline = {
    # define input and output
    'input_width': 28,
    'n_input': 784,
    'classes': [1, 4, 5, 8, 9], # [1, 4, 5, 8, 9] reduced set w/o ambiguity
    'n_train_samples_per_class': 100,
    
    # define training hyper-parameters
    'n_epochs': 10,
    'minibatch_size': 50, # default 500
    'learning_rate': 0.001, # default 0.002
    'regularization_term': 0.01, # default 0.001
    'keep_probability': 0.5, # default 0.5
    
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
    'classes': [1, 4, 5, 8, 9], # [1, 4, 5, 8, 9] reduced set w/o ambiguity
    'n_train_samples_per_class': 100,
    
    # define training hyper-parameters
    'n_epochs': 100,
    'minibatch_size': 500,
    'learning_rate': 0.002,
    'regularization_term': 0.001,
    'keep_probability': 0.5,
    
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