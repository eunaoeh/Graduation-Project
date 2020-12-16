from vgg16bn import VGG16BN

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import os
import sys
import time
import cv2
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

def cannyEdge(image):
    for i in range(0, len(image)):
#    img = utils.make_grid(image)
        img = image[i].numpy()
        img = np.transpose(img, (1, 2, 0)) 
        edge = cv2.Canny(np.uint8(img), 100, 150)
        tmp = np.tile(edge, (3, 1, 1))
        image[i] = torch.from_numpy(tmp)

def laplacianEdge(image):
    for i in range(0, len(image)):
        img = image[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        edge = cv2.Laplacian(img, cv2.CV_32F)
        image[i] = torch.from_numpy(np.transpose(edge, (2, 0, 1)))
    
def train(model, epoch):
    global not_improve_cnt, total_time


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for e in range(epoch):
        start = time.time()
        model.train()

        avg_loss = 0
        for it, (x, targets) in enumerate(train_loader):
#            cannyEdge(x)
            laplacianEdge(x)
#            print(x.size())
            x, targets = x.cuda(), targets.cuda()
            optimizer.zero_grad()

            #forward
            y = model(x)

            loss = criterion(y, targets)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
        scheduler.step()
        training_time = time.time() - start
        total_time = total_time + training_time

        print("[Epoch : {:>4d}] Average Cost : {:>.5f} Time : {:>.10f}".format(e + 1, avg_loss, training_time))
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
