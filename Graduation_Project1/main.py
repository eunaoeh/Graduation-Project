from vgg16bn import VGG16BN

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time
import numpy as np

from torchvision import datasets, transforms

torch.cuda.manual_seed_all(777)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10(root="~/.data",
                              train=True,
                              download=(not os.path.exists("~/.data")),
                              transform=train_transform)

test_data = datasets.CIFAR10(root="~/.data",
                             train=False,
                             download=(not os.path.exists("~/.data")),
                             transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=512,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=512,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=4)


best_acc = 0
not_improve_cnt = 0

total_time = 0
# tmp_total = 0
def train(model, epoch):
    global not_improve_cnt, total_time


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    arr = np.identity(32, dtype='float64')
    arr[0][0] = arr[31][31] = 0
    arr = torch.from_numpy(arr).float().to("cuda:0")
    
    for e in range(epoch):
        # start = time.time()
        model.train()
        tmp_total = 0
        avg_loss = 0
        for it, (x, targets) in enumerate(train_loader):
            x, targets = x.cuda(), targets.cuda()
            optimizer.zero_grad()

            #forward
            
            size = x.size()
            for i in range(size[0]):
                for j in range(size[1]):
                    x[i][j] = torch.mm(arr, x[i][j])
                    x[i][j] = torch.mm(x[i][j], arr)
            t = time.time()
            y = model(x)
            loss = criterion(y, targets)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
            tmp_total += time.time() - t

        scheduler.step()
        # training_time = time.time() - start
        total_time = total_time + tmp_total

        print("[Epoch : {:>4d}] Average Cost : {:>.5f} Time : {:>.10f}".format(e + 1, avg_loss, tmp_total))
        test(model, e + 1)

        if not_improve_cnt >= 100:
            break


def test(model, epoch):
    global best_acc, not_improve_cnt

    model.eval()

    correct = 0
    for x, targets in test_loader:
        x, targets = x.cuda(), targets.cuda()

        y = model(x)

        pred = y.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)

    print("Test Accuracy : {:>.2f}".format(acc))

    if acc > best_acc:
        best_acc = acc
        not_improve_cnt = 0
    else:
        not_improve_cnt += 1


if __name__ == "__main__":

    epoch = 100

    model = VGG16BN()
    model = nn.DataParallel(model)
    model = model.cuda()
    train(model, epoch)
    print("Best Accuracy : {:>.5f}".format(best_acc))
    print("Average Training Time : {:>.10f}".format(total_time/epoch))

    f = open('time.txt', "a")
    f.write("1 : ({:>.5f}, {:>.10f})".format(best_acc, total_time/epoch))
    f.close()
