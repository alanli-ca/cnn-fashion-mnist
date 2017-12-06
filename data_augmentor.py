'''
this module contains various functions to augment a (n x m) array of images
'''
import numpy as np
from skimage import transform
from skimage import filters
from skimage import util
import random
import math
import tensorflow as tf
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

def plot_images_side_by_side(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale.
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. None
    '''
    class_images = class_images.reshape(class_images.shape[0], -1)
    w_dim = int(math.sqrt(class_images.shape[-1]))
    
    plt.figure(figsize=(20,5))
    img = class_images.reshape(-1, w_dim, w_dim)
    all_concat = np.concatenate(img, 1)
    plt.imshow(all_concat, cmap='Greys')
    plt.axis('off')
    plt.show()

def flatten_dataset(dataset):
    return dataset.reshape(dataset.shape[0], -1)

def data_preprocessing_augmentor(dataset, random_shift=True, random_rotation=True,\
                                random_shear=True, random_zoom=True):
    dataset = deepcopy(dataset)
    for i in range(dataset.images.shape[0]):
        if random_shift:
            dataset._images[i,:] = tf.contrib.keras.preprocessing.image.random_shift(
                dataset._images[i,:].reshape(1,28,28), 0.2, 0.2).reshape(-1,784)
        if random_rotation:    
            dataset._images[i,:] = tf.contrib.keras.preprocessing.image.random_rotation(
                dataset._images[i,:].reshape(1,28,28), 20).reshape(-1,784)
        if random_shear:
            dataset._images[i,:] = tf.contrib.keras.preprocessing.image.random_shear(
                dataset._images[i,:].reshape(1,28,28), 0.3).reshape(-1,784)
        if random_zoom:
            dataset._images[i,:] = tf.contrib.keras.preprocessing.image.random_zoom(
                dataset._images[i,:].reshape(1,28,28), (0.7, 0.9)).reshape(-1,784)
    return dataset

def gaussian_filter_augmentor(dataset):
    '''
    applies the gaussian filter to the dataset
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width, img_width))
    
    for i in range(n_samples):
        augmented_dataset[i] = filters.gaussian(dataset[i,:,:], sigma=0.5)
        
    return flatten_dataset(augmented_dataset)

def gaussian_noise_augmentor(dataset):
    '''
    adds the gaussian noise to the dataset
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width, img_width))
    
    for i in range(n_samples):
        augmented_dataset[i] = util.random_noise(dataset[i,:,:], mode='gaussian', var=0.005)
        
    return flatten_dataset(augmented_dataset)

def poisson_noise_augmentor(dataset):
    '''
    applies the possion noise to the dataset
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width, img_width))
    
    for i in range(n_samples):
        augmented_dataset[i] = util.random_noise(dataset[i,:,:], mode='poisson')
        
    return flatten_dataset(augmented_dataset)

def fliplr_augmentor(dataset):
    '''
    flips the dataset along the vertical axis
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width, img_width))
    
    for i in range(n_samples):
        augmented_dataset[i] = np.fliplr(dataset[i,:,:])
        
    return flatten_dataset(augmented_dataset)

def swirl_rotate_augmentor(dataset):
    '''
    applies random swirl, followed by random rotation
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width, img_width))
    
    for i in range(n_samples):
        augmented_dataset[i] = transform.swirl(dataset[i,:,:], strength=random.uniform(-0.4,0.4),
                                               rotation=random.uniform(-0.4,0.4))
        
    return flatten_dataset(augmented_dataset)

def rescale_augmentor(dataset, scale_factor=2):
    '''
    rescales the dataset according to the scale factor
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m*scale_factor^2) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width*scale_factor, img_width*scale_factor))
    
    for i in range(n_samples):
        augmented_dataset[i] = transform.rescale(dataset[i,:,:], scale=scale_factor)
        
    return flatten_dataset(augmented_dataset)

def pyramid_expand_augmentor(dataset, scale_factor=2):
    '''
    rescales the dataset accroding to the scale factor, then smooths the image
    inputs:
        1. (n x m) n samples of images, each with m features
    output:
        1. (n x m*scale_factor^2) n samples of images, each with m features, after applying the augmentation
    '''
    img_width = int(math.sqrt(dataset.shape[-1]))
    dataset = dataset.reshape(-1, img_width, img_width)
    n_samples = dataset.shape[0]
    augmented_dataset = np.zeros((n_samples, img_width*scale_factor, img_width*scale_factor))
    
    for i in range(n_samples):
        augmented_dataset[i] = transform.pyramid_expand(dataset[i,:,:], upscale=scale_factor) 
        
    return flatten_dataset(augmented_dataset)