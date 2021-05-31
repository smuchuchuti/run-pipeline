#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    """Logs data to Tensorboard

    Arguments:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss",                    loss, counter)
    writer.add_figure("Image Data",        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",        mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map",        mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

def save_numpy_as_image(arr, path):
    """
    This saves image (2D array) as a file using matplotlib

    Arguments:
        arr {array} -- 2D array of pixels
        path {string} -- path to file
    """
    plt.imshow(arr, cmap="gray") #Needs to be in row,col order
    plt.savefig(path)

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape

    Returns:
        3D array of desired shape, padded with zeroes
    """

    reshaped_image = np.zeros(new_shape)

    # TASK: write your original image into the reshaped image
    # <CODE GOES HERE>
    
    reshaped_image[:image.shape[0],:image.shape[1],:image.shape[2]]=image

    return reshaped_image

