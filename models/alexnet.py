import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, input_channels, classes):
        super(AlexNet, self).__init__()

        self.c1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1,
                            padding=2)
        self.c2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        # nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            # nn.Dropout(global_params.dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(global_params.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        for i in range(1, 6):
            cl = getattr(self, 'c{}'.format(i))
            x = cl(x)
            if i in [2, 3, 5]:
                x = nn.functional.max_pool2d(x, kernel_size=2)
            if i < 5:
                x = torch.relu(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

