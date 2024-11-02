# encoding:utf-8
import os
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def file_loader(config):
    extension = config['data_ext']
    if extension == '.mat':
        return MatLoader(config)
    else:
        return OtherLoader(config)


def image_loader(path):
    return Image.open(path).convert('L')



class MatLoader:
    def __init__(self, config):
        norm_input = config['norm_input']
        norm_label = config['norm_label']
        self.norm_flag = config['norm_flag']
        self.min_max_list = [norm_input, norm_label]

    def array_norm(self, data, index):
        if self.norm_flag is None:
            return data

        min_max = self.min_max_list[index]

        if min_max is None:
            return data

        if None in min_max:
            if min_max[0] != min_max[1]:
                return data
        elif min_max[0] >= min_max[1]:
            return data

        if (min_max[0] is None) and (min_max[1] is None):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        _range = min_max[1] - min_max[0]
        return (data - min_max[0]) / _range

    def __call__(self, path, index, **kwargs):
        ret = sio.loadmat(path)
        for item in ret.values():
            if isinstance(item, np.ndarray):
                return Image.fromarray(self.array_norm(item, index))


class OtherLoader:
    def __init__(self, config):
        norm_input = config['norm_input']
        norm_label = config['norm_label']
        self.norm_flag = config['norm_flag']
        self.min_max_list = [norm_input, norm_label]

    def array_norm(self, data, index):
        if self.norm_flag is None:
            return data

        min_max = self.min_max_list[index]

        if min_max is None:
            return data

        if None in min_max:
            if min_max[0] != min_max[1]:
                return data
        elif min_max[0] >= min_max[1]:
            return data

        if (min_max[0] is None) and (min_max[1] is None):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        _range = min_max[1] - min_max[0]
        return (data - min_max[0]) / _range

    def __call__(self, path, index, **kwargs):
        ret = image_loader(path)
        return self.array_norm(np.asarray(ret, dtype=np.float32), index)


class MyDataSet1(Dataset):
    def __init__(self, img_file, data_root, data_size, label_size, loader):
        imgs = []
        with open(img_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                imgs.append((temp[0], temp[1]))

        self.data_root = data_root
        self.imgs = imgs
        self.data_size = data_size
        self.label_size = label_size
        self.loader = loader

    def __getitem__(self, index):
        fn1, labelfn = self.imgs[index]
        img = self.loader(os.path.join(self.data_root, fn1), 0)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)

        img = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((self.data_size[1], self.data_size[2]))])(img)

        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[1], self.label_size[2]))])(label)

        if fn1.split('\\')[1:]:
            fn1 = fn1.split('\\')[1:]
            fn = fn1[-2] + ',' + fn1[-1]
        else:
            fn1 = fn1.split('/')[1:]
            fn = fn1[0] + ',' + fn1[1]
        return img, label, fn

    def __len__(self):
        return len(self.imgs)


class MyDataSet2(Dataset):
    def __init__(self, img_file, data_root, data_size, label_size, loader):
        imgs = []
        with open(img_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                imgs.append((temp[0], temp[1], temp[2]))

        self.data_root = data_root
        self.imgs = imgs
        self.data_size = data_size
        self.label_size = label_size
        self.loader = loader

    def __getitem__(self, index):
        fn1, labelfn, edgefn = self.imgs[index]
        img = self.loader(os.path.join(self.data_root, fn1), 0)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)
        edge = self.loader(os.path.join(self.data_root, edgefn), 1)

        img = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((self.data_size[1], self.data_size[2]))])(img)
        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[1], self.label_size[2]))])(label)
        edge = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((self.label_size[1], self.label_size[2]))])(edge)

        if fn1.split('\\')[1:]:
            fn1 = fn1.split('\\')[1:]
            fn = fn1[0] + ',' + fn1[1]
        else:
            fn1 = fn1.split('/')[1:]
            fn = fn1[0] + ',' + fn1[1]

        return [img], [label, edge], fn

    def __len__(self):
        return len(self.imgs)


def dataloader(config):

    trainset = MyDataSet1(config['train_db'], config['db_root'],
                          config['data_size'], config['label_size'], loader=file_loader(config))


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size_train'],
                                              shuffle=False, num_workers=0)

    testset = MyDataSet1(config['test_db'], config['db_root'],
                         config['data_size'], config['label_size'], loader=file_loader(config))


    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size_test'],
                                             shuffle=False, num_workers=0)

    return trainloader, testloader


if __name__ == '__main__':
    import json
    with open(r'\MyNet\config.json', 'r') as f1:
        config1 = json.load(f1)
    trainset1 = MyDataSet1('H:/MyNet1110/all_image/label_train_1.txt', 'H:/MyNet1110/all_image',
                           config1['data_size'], config1['label_size'], loader=file_loader(config1))
    print(trainset1.__dir__())
    print(iter(trainset1).__dir__())

    a = range(5)
    print(a.__dir__())
    print(iter(a).__dir__())

    from collections.abc import Iterable, Iterator
    print(isinstance(trainset1, Iterable), isinstance(trainset1, Iterator), sep=' ')
