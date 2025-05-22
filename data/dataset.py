import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import subprocess
import os
import torchvision

data_dir = './data'
mean = [0.5]
std = [0.5]


class dataset_transform(Dataset):
    def __init__(self, dataset, select_classes=None, target_transform=None):
        self.dataset = dataset
        self.target_transform = target_transform
        if select_classes is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = [idx for idx, (_, y) in enumerate(dataset) if y in select_classes]
            self.transform_dict = dict(zip(select_classes, list(range(len(select_classes)))))
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]  
        label = self.dataset[self.indices[idx]][1]
        if self.target_transform=='reindex':
            label = self.transform_dict[label]
        elif self.target_transform=='open':
            label = 999
        else:
            label = label
        return (image, label)
    def __len__(self):
        return len(self.indices)


class OPENWORLDmal(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDmal, self).__init__(root, train, transform, target_transform, download=False)

        if train:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/mal_32_1c_train.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/mal_32_1c_test.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)


class combined_USTC_mal(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(combined_USTC_mal, self).__init__(root, train, transform, target_transform, download=False)
        if train:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/combined_train_data.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/combined_test_data.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)

class USTC(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(USTC, self).__init__(root, train, transform, target_transform, download=False)

        if train:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/USTC_1c_train.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)
        else:
            loaded_data = np.load(r'/home/ju/Desktop/NetMamba/MGPL/data/mamba数据/USTC_1c_test.npz')
            self.data = loaded_data['data']
            self.targets = loaded_data['target']
            self.data = np.vstack(self.data).reshape(-1, 32, 32)


def get_dataset(dataset, train=False, select_classes=None, target_transform=None):
    if dataset == 'mal':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = OPENWORLDmal(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = OPENWORLDmal(root=data_dir, train=False, transform=transform_test, download=False)
    
    elif dataset == 'combined_USTC_mal':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = combined_USTC_mal(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = combined_USTC_mal(root=data_dir, train=False, transform=transform_test, download=False)
    
    elif dataset == 'USTC':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = USTC(root=data_dir, train=True, transform=transform_train, download=False)
        test_set = USTC(root=data_dir, train=False, transform=transform_test, download=False)
    
    else:
        raise ValueError('Unsupported dataset: ' + dataset)
    
    if train:
        return dataset_transform(train_set, select_classes, target_transform)
    else:
        return dataset_transform(test_set, select_classes, target_transform)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]
    total_classes = list(range(10))
    unknown_classes = splits[0]
    known_classes = list(set(total_classes) - set(unknown_classes))

    known_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6]
    train_set = get_dataset('mal', True, known_classes, 'reindex')
    test_set = get_dataset('mal', False, known_classes, 'reindex')
    open_set = get_dataset('mal', False, unknown_classes, 'open')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, drop_last = True)

