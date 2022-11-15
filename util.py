"""Utilities for scoring the model."""
import torch


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()

def increase_image_channels(images, num_out_channels):
    """Updates an image with updated number of channels to feed into a pretrained model

    Args:
        image (torch.Tensor): batch image
            shape (B, C, H, W)
        num_out_channels: int
    """
    temp = torch.empty((images.size(0), num_out_channels, images.size(2), images.size(3)))

    image_mean = torch.mean(images, axis = 1, keepdim=True)
    for i in range(num_out_channels):
        if i < images.size(1):
            temp[:, i, :, :] = images[:, i, :, :]
        else:
            temp[:, i, :, :] = image_mean
    images = temp
    return

    