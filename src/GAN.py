import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)

class Music2ImageGenerator(nn.Module):
    def __init__(self, z_dim=512, embed_size=512, channels=3, conv_dim=64):
        super().__init__()
        self.l1 = conv_block(z_dim + embed_size, conv_dim * 8, pad=0, transpose=True)
        self.l2 = conv_block(conv_dim * 8, conv_dim * 4, transpose=True)
        self.l3 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.l4 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.l5 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, music):
        x = torch.cat([x, music], dim=1).reshape(x.shape[0], -1, 1, 1)
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        x = torch.tanh(self.l5(x))
        return x


class FakeImageDiscriminator(nn.Module):
    def __init__(self, channels=3, conv_dim=64):
        super().__init__()
        self.l1 = conv_block(channels, conv_dim, use_bn=False)
        self.l2 = conv_block(conv_dim, conv_dim * 2)
        self.l3 = conv_block(conv_dim * 2, conv_dim * 4)
        self.l4 = conv_block(conv_dim * 4, conv_dim * 8)
        self.l5 = conv_block(conv_dim * 8, conv_dim * 8, k_size=4, stride=1, pad=0, use_bn=False)
        self.fc = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        x = x.squeeze()
        x = self.fc(x)
        return F.sigmoid(x)