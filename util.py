"""Utilities for scoring the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def increase_image_channels(images, num_out_channels, device):
    """Updates an image with updated number of channels to feed into a pretrained model

    Args:
        image (torch.Tensor): batch image
            shape (B, C, H, W)
        num_out_channels: int
    """
    temp = torch.empty((images.size(0), num_out_channels, images.size(2), images.size(3)))

    image_mean = torch.mean(images, axis = 1)
    for i in range(num_out_channels):
        if i < images.size(1):
            temp[:, i, :, :] = images[:, i, :, :]
        else:
            temp[:, i, :, :] = image_mean
    
    return temp.to(device)



class aug_net_block(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size
    ):
        """Inits the augmentation network for MetaAugNet on MAML"""
        super(aug_net_block, self).__init__()

        self.conv_param = nn.init.constant_(
                    torch.empty(
                        out_channel,
                        in_channel,
                        kernel_size,
                        kernel_size,
                        requires_grad=True
                    ),
                    0
                )
        self.conv_bias = nn.init.zeros_(
                    torch.empty(
                        out_channel,
                        requires_grad=True
                    )
                )
        self.conv_identity_weight = nn.init.dirac_(
            torch.empty(
                out_channel, 
                in_channel, 
                kernel_size, 
                kernel_size, 
                requires_grad = False)
                )

    def forward(self, x):
        """x: input image (B, C, H, W)"""
        res =  F.conv2d(input = x, weight = self.conv_identity_weight, bias = None, padding = 'same', stride = 1)
        x = F.conv2d(
            input = x,
            weight = self.conv_param,
            bias = self.conv_bias,
            stride = 1,
            padding = 'same'
        )
        x = torch.max(x, 0)
        return x + res

class mean_pool_along_channel(nn.Module):
    def __init__(self):
        super(mean_pool_along_channel, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 4
        return torch.mean(x, dim = [2,3])


class manual_relu(nn.Module):

    def __init__(
        self
    ):

        super(manual_relu, self).__init__()

    def forward(self, x):
        return torch.max(x, 0)



    