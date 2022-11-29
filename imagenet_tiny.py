"""Dataloading for Imagenet Tiny."""
import os
import glob
import gzip

from kaggle.api.kaggle_api_extended import KaggleApi
import imageio
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import sys
from PIL import Image

NUM_TRAIN_CLASSES = 64
NUM_VAL_CLASSES = 16
NUM_TEST_CLASSES = 20
NUM_SAMPLES_PER_CLASS = 600


def load_image(file_path):
    """Loads and transforms an imagenet tiny image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    # x = imageio.imread(file_path)
    # if len(x.shape) != 3:
    #     x = np.repeat(x[:, :, np.newaxis], 3, -1)
    # assert len(x.shape) == 3
    # x = np.moveaxis(x, 2, 0)
    # x = x.copy()

    x = Image.open(file_path).convert("RGB")

    std_image = Compose(
            [
                Resize(
                    size=(48, 48)
                ),        
                ToTensor(),
                Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )
    x = std_image(x)
    return x

class ImagenetDataset(dataset.Dataset):
    """imagenet dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _MAIN_IMAGENET_PATH = './ILSVRC/Data/CLS-LOC/train'
    _MINY_IMAGENET_PATH = './mini-imagenet-tools/processed_images'
    def __init__(self, num_support, num_query):
        """Inits ImagenetDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # if necessary, download the main imagenet dataset
        if not os.path.isdir(self._MAIN_IMAGENET_PATH) and not os.path.isdir(self._MINY_IMAGENET_PATH):
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files('imagenet-object-localization-challenge', path='./')
            
        if not os.path.isdir(self._MINY_IMAGENET_PATH)
            raise Exception("Download and process MINY via mini-imagenet-toolkit")


        # get all image folders
        self._image_folders = glob.glob(
            os.path.join(self._MINY_IMAGENET_PATH, '*/'))
        assert len(self._image_folders) == (
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )

        # shuffle images
        # np.random.default_rng(0).shuffle(self._image_folders)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._image_folders[class_idx], '*.JPEG')
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class ImagenetSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_imagenet_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=ImagenetDataset(num_support, num_query),
        batch_size=batch_size,
        sampler=ImagenetSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
