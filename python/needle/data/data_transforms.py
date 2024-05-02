import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        assert len(img.shape) == 3
        if flip_img:
            img = img[:,::-1,:]
        
        return img        
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        shape = img.shape
        assert len(shape) == 3
        img = np.pad(img, 
                     pad_width=[(self.padding, self.padding), (self.padding, self.padding), (0, 0)],
                     mode='constant',
                     constant_values=0)
        img = img[self.padding + shift_x:self.padding + shift_x + shape[0],
                  self.padding + shift_y:self.padding + shift_y + shape[1],
                  :]
        return img
        ### END YOUR SOLUTION
