import torch
import torch.nn as nn


class VGG16BN(nn.Module):

    def __init__(self):
        super(VGG16BN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(),
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(),
        )

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(512, 10)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.linear2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
