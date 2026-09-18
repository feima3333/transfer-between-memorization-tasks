"""The paper's architecture: a 6-conv + FC CNN with BatchNorm (WideCNN_BN).

Six 3x3 conv layers in three width-(128, 256, 512) blocks -- each block is two ConvBNReLU
plus a 2x2 max-pool -- then global average pooling and a linear readout. Conv weights use
Kaiming-normal init; the FC readout starts deliberately small at N(0, 0.01). That small FC
init is load-bearing, not a detail: it is why the FC norm grows ~20x over training (Table 1)
and motivates the FC-rescale control, so do not "tidy" it to a default initializer.
"""
import torch.nn as nn

from common.config import NUM_CLASSES, WIDTHS


class ConvBNReLU(nn.Module):
    """3x3 conv (bias=True) -> BatchNorm -> ReLU.

    Conv keeps bias=True even though the following BatchNorm could absorb a bias: the SVD /
    alignment surgery (Fig 5) treats [conv.weight; conv.bias] as one block, so the bias is
    part of the object being measured and must exist.
    """

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class WideCNN_BN(nn.Module):
    """6-conv + FC CNN with BatchNorm; widths default to (128, 256, 512)."""

    def __init__(self, num_classes=NUM_CLASSES, widths=WIDTHS):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(ConvBNReLU(3, w1), ConvBNReLU(w1, w1), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(ConvBNReLU(w1, w2), ConvBNReLU(w2, w2), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(ConvBNReLU(w2, w3), ConvBNReLU(w3, w3), nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(w3, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # small FC init -- see module docstring
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=(2, 3))  # global average pool
        x = self.fc(x)
        return x
