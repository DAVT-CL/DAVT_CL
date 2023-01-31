import torch
import math
import sys
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import relu

# ResNet18
class Bottleneck(nn.Module):
    expansion = 4  # # output cahnnels / # input channels

    def __init__(self, inplanes, outplanes, stride=1):
        assert outplanes % self.expansion == 0
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.bottleneck_planes = int(outplanes / self.expansion)
        self.stride = stride

        self._make_layer()

    def _make_layer(self):
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(self.inplanes, self.bottleneck_planes, kernel_size=1, stride=self.stride, bias=False)
        # conv 3x3
        self.bn2 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv2 = nn.Conv2d(self.bottleneck_planes, self.bottleneck_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # conv 1x1
        self.bn3 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv3 = nn.Conv2d(self.bottleneck_planes, self.outplanes, kernel_size=1,
                               stride=1)
        if self.inplanes != self.outplanes:
            self.shortcut = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1,
                                      stride=self.stride, bias=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out


class ResNet164(nn.Module):
    def __init__(self):
        super(ResNet164, self).__init__()
        nstages = [16, 64, 128, 256]
        # one conv at the beginning (spatial size: 32x32)
        self.conv1 = nn.Conv2d(3, nstages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        depth = 164
        block = Bottleneck
        n = int((depth - 2) / 9)
        # use `block` as unit to construct res-net
        # Stage 0 (spatial size: 32x32)
        self.layer1 = self._make_layer(block, nstages[0], nstages[1], n)
        # Stage 1 (spatial size: 32x32)
        self.layer2 = self._make_layer(block, nstages[1], nstages[2], n, stride=2)
        # Stage 2 (spatial size: 16x16)
        self.layer3 = self._make_layer(block, nstages[2], nstages[3], n, stride=2)
        # Stage 3 (spatial size: 8x8)
        self.bn = nn.BatchNorm2d(nstages[3])
        self.relu = nn.ReLU(inplace=True)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, nstage, stride=1):
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        for i in range(1, nstage):
            layers.append(block(outplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
####################################################################################################


# ResNet18
def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out, inplace=True)
        return out


class ResNet18Pre(nn.Module):
    def __init__(self, nf=64, stages=3):
        super(ResNet18Pre, self).__init__()
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 3, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, nf * 4, num_blocks[3], stride=2)
        self._resnet_high = nn.Sequential(
                                          self.layer4,
                                          nn.Identity()
                                          )
        if nf == 64:
            if self.stages == 3:
                self.resnet_low = nn.Sequential(self.conv1,
                                                self.bn1,
                                                self.relu,
                                                self.layer1,  # 64, 32, 32
                                                self.layer2,  # 128, 16, 16
                                                # self.layer3,  # 256, 8, 8
                                                )
            if self.stages == 2:
                self.resnet_low = nn.Sequential(self.conv1,
                                                self.bn1,
                                                self.relu,
                                                self.layer1,  # 64, 32, 32
                                                self.layer2,  # 128, 16, 16
                                                self.layer3,  # 256, 8, 8
                                                )
            if self.stages == 1:
                self.resnet_low = nn.Sequential(self.conv1,
                                                self.bn1,
                                                self.relu,
                                                self.layer1,  # nf, h, w
                                                self.layer2,  # 2*nf, h/2, w/2
                                                self.layer3,  # 4*nf, h/4, w/4
                                                # self.layer4  # 8*nf, h/8, w/8
                                                )


        else:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # nf, h, w
                                            self.layer2,  # 2*nf, h/2, w/2
                                            self.layer3,  # 4*nf, h/4, w/4
                                            # self.layer4  # 8*nf, h/8, w/8
                                            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class ResNet18Pre224(nn.Module):
    def __init__(self, stages):
        super(ResNet18Pre224, self).__init__()

        nf = 64
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2)
        # self._resnet_high = nn.Sequential(
        #                                   self.layer4,
        #                                   nn.Identity()
        #                                   )
        if self.stages == 2 or self.stages == 1:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.maxpool,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # 64, 32, 32
                                            self.layer2,  # 128, 16, 16
                                            self.layer3,  # 256, 8, 8
                                            # self.layer4
                                            )
        if self.stages == 3:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.maxpool,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # 64, 32, 32
                                            self.layer2,  # 128, 16, 16
                                            # self.layer3,  # 256, 8, 8
                                            # self.layer4
                                            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
