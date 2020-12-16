from vgg16bn import VGG16BN

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import os
import sys

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


def train(model, epoch):
    global not_improve_cnt

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for e in range(epoch):
        model.train()

        avg_loss = 0
        for it, (x, targets) in enumerate(train_loader):
            x, targets = x.cuda(), targets.cuda()
            optimizer.zero_grad()

            #forward
            y = model(x)

            loss = criterion(y, targets)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)
        scheduler.step()

        print("[Epoch : {:>4d}] Average Cost : {:>.5f}".format(e + 1, avg_loss))
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
