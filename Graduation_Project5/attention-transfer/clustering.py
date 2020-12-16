import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils


test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
train_transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            test_transform
        ])

train_data = datasets.CIFAR10(root="~/.data",
                              train=True,
                              download=(not os.path.exists("~/.data")),
                              transform=train_transform)


test_data = datasets.CIFAR10(root="~/.data/cifar-10-batches-py/data_batch_1",
                             train=False,
                             download=(not os.path.exists("~/.data")),
                             transform=test_transform)

def main():
    #_tasks = transforms.Compose([transforms.ToTensor()])
    #mnist_trainset = datasets.MNIST(root=’./data’, train=True, download=True, transform=_tasks)
    #print(mnist_trainset)
    # kmeans = KMeans(n_clusters=10)
    # KMmodel = kmeans.fit(mnist_trainset)

    def create_iterator(data, mode):
        return DataLoader(data, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())
   
    train_loader = create_iterator(train_data, True) # True 이면 train 아니면 test 해서 필요한? 셋?? 설정
    test_loader = create_iterator(test_data, False)

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # count = [0]*10

    # for i in range(5000):
    #     count[labels[i]] += 1   
    # print(count)

if __name__ == '__main__':
    main()
