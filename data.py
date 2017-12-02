import numpy as np
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(42)

def load_data_as_array(one_hot=False):
    '''
    loads and transforms the training and test data and labels as a numpy array
    
    input: none
    output: training images, training labels, test images, test labels - all as numpy array
    '''
    data = MNIST('./data/fashion')
    train_images, train_labels = data.load_training()
    train_images, train_labels = np.asarray(train_images, dtype=np.int32), np.asarray(train_labels, dtype=np.int32)
    test_images, test_labels = data.load_testing()    
    test_images, test_labels = np.asarray(test_images, dtype=np.int32), np.asarray(test_labels, dtype=np.int32)
    
    if one_hot==True:
        label_binarizer = preprocessing.LabelBinarizer()
        label_binarizer.fit(np.arange(10))
        train_labels_one_hot = label_binarizer.transform(train_labels)
        test_labels_one_hot = label_binarizer.transform(test_labels)
        return train_images, train_labels_one_hot, test_images, test_labels_one_hot
    else:
        return train_images, train_labels, test_images, test_labels

def load_data(n_samples_per_class=100):
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)
    train_data = fashion_mnist.train
    valid_data = fashion_mnist.validation
    test_data = fashion_mnist.test
    
    subset_idx = np.array([])
    class_labels = np.argmax(train_data.labels, axis=1)
    for i in range(10):
        class_idx = np.squeeze(np.argwhere(class_labels==i))
        np.random.shuffle(class_idx)
        class_idx = class_idx[0:n_samples_per_class]
        subset_idx = np.concatenate((subset_idx, class_idx)).astype(int)
    np.random.shuffle(subset_idx)
    train_data._images = train_data._images[subset_idx]
    train_data._labels = train_data._labels[subset_idx]
    train_data._num_examples = n_samples_per_class * 10

    return train_data, valid_data, test_data
    
def get_class_labels():
    '''
    Maps numerical labels to text labels
    
    input: none
    output: dictionary mapping the numerical class label to the text class label
    '''
    class_labels = {
        0: "T-shirt/top",
        1: "Pants",
        2: "Sweater",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Boot"
    }
    
    return class_labels