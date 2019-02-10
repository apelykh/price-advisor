from torchvision import transforms


class Resize:
    """
    Resize the image in a sample to a given size.

    :param output_size (tuple or int): Desired output size. If tuple,
        output is matched to output_size. If int, smaller of image
        edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        resize = transforms.Resize(self.output_size)

        return {
            'image': resize(sample['image']),
            'category': sample['category'],
            'condition': sample['condition']
        }


class CenterCrop:
    """
    Crop the sample at the center.

    :param output_size (tuple or int): Desired output size of the crop.
        If size is an int, a square crop (output_size, output_size) is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        center_crop = transforms.CenterCrop(self.output_size)

        return {
            'image': center_crop(sample['image']),
            'category': sample['category'],
            'condition': sample['condition']
        }


class Normalize:
    """
    Normalize a sample with mean and standard deviation.

    :param mean: Sequence of means for each channel.
    :param std: Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        normalize = transforms.Normalize(self.mean, self.std)

        return {
            'image': normalize(sample['image']),
            'category': sample['category'],
            'condition': sample['condition']
        }


class ToTensor:
    """
    Convert fields of sample to Tensors.
    """

    def __call__(self, sample):
        to_tensor = transforms.ToTensor()

        return {
            'image': to_tensor(sample['image']),
            'category': sample['category'],
            'condition': sample['condition']
        }
