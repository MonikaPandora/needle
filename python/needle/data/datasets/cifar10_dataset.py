import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        import pickle
        import os
        self.p = p
        self.transforms = transforms
        if train:
            Xs = []
            ys = []
            for i in range(1, 6):
                with open(os.path.join(base_folder, f'data_batch_{i}'), 'rb') as f:
                    datas = pickle.load(f, encoding='bytes')
                    Xs.append(datas[b'data'])
                    ys += datas[b'labels']
            self.X = np.concatenate(Xs, axis=0).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)) / 255.
            self.y = np.array(ys, dtype=np.uint8)
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as f:
                datas = pickle.load(f, encoding='bytes')
            self.X = datas[b'data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)) / 255.
            self.y = np.array(datas[b'labels'], dtype=np.uint8)
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        if len(imgs.shape) == 4:
            if np.random.rand() < self.p:
                for b in range(imgs.shape[0]):
                    imgs[b] = self.apply_transforms(imgs[b])
            return imgs.transpose((0, 3, 1, 2)), self.y[index]
        else:
            if np.random.rand() < self.p:
                imgs = self.apply_transforms(imgs)
            return imgs.transpose((2, 0, 1)), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION
