import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_epoch_errors(train_errors, valid_errors, prefix="temp"):
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(train_errors, label="train")
    ax1.plot(valid_errors, label="valid", linestyle="dashed")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error")
    ax1.set_title("Error vs. Epoch")
    ax1.set_ylim([0, 1])
    ax1.legend()

    plt.tight_layout()
    plt.savefig("./images/{}_error_vs_epoch.png".format(prefix))
    plt.show()

def plot_train_iteration_errors(train_iter_errors, prefix="temp"):
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(train_iter_errors, label="train")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Error")
    ax1.set_title("Error vs. Iteration")
    ax1.set_ylim([0, 1])
    ax1.legend()

    plt.tight_layout()
    plt.savefig("./images/{}_error_vs_iteration.png".format(prefix))
    plt.show()
    
def plot_cnn_kernels(vis_layers, W_conv1, prefix="temp"):
    fig = plt.figure(figsize=(vis_layers.shape[0], 1))
    for i in range(vis_layers.shape[0]):
        ax = fig.add_subplot(1, vis_layers.shape[0], i+1)
        ax.imshow(tf.squeeze(W_conv1)[:,:,vis_layers[i]].eval(), cmap='gray')
        ax.axis("off")
        
    plt.savefig("./images/{}_filter_visualization.png".format(prefix))
    plt.show()

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