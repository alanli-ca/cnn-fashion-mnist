import numpy as np
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
import data_augmentor
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

def load_data(n_train_samples_per_class=100):
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True)
    train_data = fashion_mnist.train
    valid_data = fashion_mnist.validation
    test_data = fashion_mnist.test
    
    subset_idx = np.array([])
    class_labels = np.argmax(train_data.labels, axis=1)
    for i in range(10):
        class_idx = np.squeeze(np.argwhere(class_labels==i))
        np.random.shuffle(class_idx)
        class_idx = class_idx[0:n_train_samples_per_class]
        subset_idx = np.concatenate((subset_idx, class_idx)).astype(int)
    np.random.shuffle(subset_idx)
    train_data._images = train_data._images[subset_idx]
    train_data._labels = train_data._labels[subset_idx]
    train_data._num_examples = n_train_samples_per_class * 10

    return train_data, valid_data, test_data

def augment_data(dataset, augment_type=None):
    if augment_type is None:
        raise Exception("augment_type is not provided")
    elif augment_type == "GAUSSIAN_FILTER":
        return data_augmentor.gaussian_filter_augmentor(dataset)
    elif augment_type == "GAUSSIAN_NOISE":
        return data_augmentor.gaussian_noise_augmentor(dataset)
    elif augment_type == "POISSON_NOISE":
        return data_augmentor.poisson_noise_augmentor(dataset)
    elif augment_type == "FLIP_LR":
        return data_augmentor.fliplr_augmentor(dataset)
    elif augment_type == "SWIRL_ROTATE":
        return data_augmentor.swirl_rotate_augmentor(dataset)
    elif augment_type == "SCALE_UP":
        return data_augmentor.rescale_augmentor(dataset)
    elif augment_type == "PYRAMID_EXPAND":
        return data_augmentor.pyramid_expand_augmentor(dataset)
    else:
        raise Exception("augment_type must be one of: GAUSSIAN_FILTER, GAUSSIAN_NOISE " +
        "POISSON_NOISE, FLIP_LR, SWIRL_ROTATE, PYRAMID_EXPAND")
    
    
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