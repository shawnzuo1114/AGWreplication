import torch
import torch.nn as nn


class conv_a(nn.Module):
    def __init__(self):
        super(conv_a, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class gm_pool(nn.Module):
    def __init__(self, p_init=3, eps=1e-6):
        super(gm_pool, self).__init__()
        self.p = nn.Parameter(torch.tensor(p_init, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return torch.clamp(x, min=self.eps).pow(self.p).mean(dim=(-1, -2), keepdim=True).pow(1.0 / self.p)

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        # 官方逻辑中 inter_channels 计算有问题，这里改为 in_channels // reduc_ratio
        self.inter_channels = in_channels // reduc_ratio
        if self.inter_channels == 0:
            self.inter_channels = 1

        # g(x) 变换
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # W 变换 + BN
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        # 初始化 BN 权重为 0
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        # θ 和 φ 变换
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x: B x C x H x W
        """
        batch, C, H, W = x.size()

        # g(x)
        g_x = self.g(x).view(batch, self.inter_channels, -1)  # B x C' x N
        g_x = g_x.permute(0, 2, 1)  # B x N x C'

        # θ(x)
        theta_x = self.theta(x).view(batch, self.inter_channels, -1)  # B x C' x N
        theta_x = theta_x.permute(0, 2, 1)  # B x N x C'

        # φ(x)
        phi_x = self.phi(x).view(batch, self.inter_channels, -1)  # B x C' x N

        # 相似度矩阵
        f = torch.matmul(theta_x, phi_x)  # B x N x N
        f_div_C = torch.softmax(f, dim=-1) # 归一化

        # 输出
        y = torch.matmul(f_div_C, g_x)  # B x N x C'
        y = y.permute(0, 2, 1).contiguous().view(batch, self.inter_channels, H, W)  # B x C' x H x W

        # W 变换 + BN + 残差
        W_y = self.W(y)
        z = W_y + x

        return z

class agw(nn.Module):
    def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):
        super(agw, self).__init__()
        self.inplanes = 64
        self.conv1 = conv_a()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 最后一层支持自定义 stride 和 dilation
        self.nl = NonLocalBlock(256 * block.expansion, reduc_ratio=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)
        self.gmpool = gm_pool()
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.bn2.bias.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.nl(x)
        x = self.layer4(x)
        x = self.gmpool(x)  # 输出 (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        feat = self.bn2(x)
        return feat


# 示例：构建 AGW 网络（类似 ResNet50）
# layers参数对应每一层 Bottleneck 的块数，[3,4,6,3] 对应 ResNet50
model = agw(Bottleneck, layers=[3, 4, 6, 3])
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(out)  # 输出应为 (2, 2048, 1, 1)
