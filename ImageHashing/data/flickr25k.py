import torch
import numpy as np
import os

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Loading nus-wide dataset.

    Args:
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader (torch.evaluate.data.DataLoader): Data loader.
    """

    Flickr25k.init(root, num_query, num_train)
    query_dataset = Flickr25k(root, 'query', query_transform())
    train_dataset = Flickr25k(root, 'train', train_transform())
    retrieval_dataset = Flickr25k(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class Flickr25k(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = Flickr25k.TRAIN_DATA
            self.targets = Flickr25k.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Flickr25k.QUERY_DATA
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Flickr25k.RETRIEVAL_DATA
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        """
        Initialize dataset

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        # Load dataset
        img_txt_path = os.path.join(root, 'img.txt')
        targets_txt_path = os.path.join(root, 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        Flickr25k.QUERY_DATA = data[query_index]
        Flickr25k.QUERY_TARGETS = targets[query_index, :]

        Flickr25k.TRAIN_DATA = data[train_index]
        Flickr25k.TRAIN_TARGETS = targets[train_index, :]

        Flickr25k.RETRIEVAL_DATA = data[retrieval_index]
        Flickr25k.RETRIEVAL_TARGETS = targets[retrieval_index, :]
