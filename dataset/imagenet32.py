from torchvision.datasets import CIFAR10
class Imagenet32(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'imagenet32'
    url = "https://image-net.org/data/downsample/Imagenet32_train.zip"
    filename = "Imagenet32_train.zip"
    train_list = [['train_data_batch_%s'%i, None] for i in range(1,11)]

    test_list = [
        ['val_data', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    def check_integrity(self, path, md5):
        return True
    def _check_integrity(self):
        return True
    def _load_meta(self):
        pass
    def __getitem__(self, index: int):
        img, target =  super().__getitem__(index)
        return img, target -1 