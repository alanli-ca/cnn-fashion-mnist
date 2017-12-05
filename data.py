import numpy as np
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import data_augmentor
from scipy.ndimage import imread
from sklearn import preprocessing
np.random.seed(42)
tf.set_random_seed(42)

def load_data(n_train_samples_per_class=100, classes=np.arange(10)):
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True, seed=42)
    train_data = fashion_mnist.train
    valid_data = fashion_mnist.validation
    test_data = fashion_mnist.test
    classes = classes
    labels_keep = np.asarray([x for x in range(10) if x not in classes])
    
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

    train_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(train_data.labels, axis=1), classes))).astype(int)
    valid_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(valid_data.labels, axis=1), classes))).astype(int)
    test_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(test_data.labels, axis=1), classes))).astype(int)

    train_data._images = train_data._images[train_truncated_idx]
    train_data._labels = np.delete(train_data._labels[train_truncated_idx], labels_keep, axis=1)
    train_data._num_examples = int(n_train_samples_per_class * classes.shape[0])
    valid_data._images = valid_data._images[valid_truncated_idx]
    valid_data._labels = np.delete(valid_data._labels[valid_truncated_idx], labels_keep, axis=1)
    valid_data._num_examples = int(valid_data.num_examples * classes.shape[0]/10)
    test_data._images = test_data._images[test_truncated_idx]
    test_data._labels = np.delete(test_data._labels[test_truncated_idx], labels_keep, axis=1)
    test_data._num_examples = int(test_data.num_examples * classes.shape[0]/10)
    
    return train_data, valid_data, test_data

def load_data_with_augmentation(n_train_samples_per_class=100, classes=np.arange(10),\
                                random_shift=True, random_rotation=True, random_shear=True, random_zoom=True):
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True, seed=42)
    train_data = fashion_mnist.train
    valid_data = fashion_mnist.validation
    test_data = fashion_mnist.test
    classes = classes
    labels_keep = np.asarray([x for x in range(10) if x not in classes])
    
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

    train_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(train_data.labels, axis=1), classes))).astype(int)
    valid_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(valid_data.labels, axis=1), classes))).astype(int)
    test_truncated_idx = np.squeeze(np.argwhere(np.in1d(np.argmax(test_data.labels, axis=1), classes))).astype(int)

    train_data._images = train_data._images[train_truncated_idx]
    train_data._labels = np.delete(train_data._labels[train_truncated_idx], labels_keep, axis=1)
    train_data._num_examples = int(n_train_samples_per_class * classes.shape[0])
    valid_data._images = valid_data._images[valid_truncated_idx]
    valid_data._labels = np.delete(valid_data._labels[valid_truncated_idx], labels_keep, axis=1)
    valid_data._num_examples = int(valid_data.num_examples * classes.shape[0]/10)
    test_data._images = test_data._images[test_truncated_idx]
    test_data._labels = np.delete(test_data._labels[test_truncated_idx], labels_keep, axis=1)
    test_data._num_examples = int(test_data.num_examples * classes.shape[0]/10)
    
    train_data_modified = data_augmentor.data_preprocessing_augmentor(train_data, random_shift=random_shift, random_rotation=random_rotation, random_shear=random_shear, random_zoom=random_zoom)
    valid_data_modified = data_augmentor.data_preprocessing_augmentor(valid_data, random_shift=random_shift, random_rotation=random_rotation, random_shear=random_shear, random_zoom=random_zoom)
    test_data_modified = data_augmentor.data_preprocessing_augmentor(test_data, random_shift=random_shift, random_rotation=random_rotation, random_shear=random_shear, random_zoom=random_zoom)
    return train_data_modified, valid_data_modified, test_data_modified

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
    
    
def get_class_labels(subset=False):
    '''
    Maps numerical labels to text labels
    
    input: none
    output: dictionary mapping the numerical class label to the text class label
    '''
    if subset:
        class_labels = {
            0: "Pants",
            1: "Coat",
            2: "Sandal",
            3: "Bag",
            4: "Boot"
        }
    else:
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

def load_scraped_data():
    data = np.asarray([])
    labels = np.asarray([])
    for i in range(5):
        for j in range(10):
            image = imread("./test_images_hard_bw/{}_{}.jpg".format(i, j), flatten=True).reshape(1, 784)
            data = np.append(data, image)
            labels = np.append(labels, i)
    data = (1 - data.reshape(-1, 784)/255.0)

    labels = labels.astype(int)
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.sort(labels))
    labels = lb.transform(labels)
    return data, labels