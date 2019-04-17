import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()
        self.up = upscale_factor
        
        self.conv1 = nn.Conv2d(num_channels, base_filter, 
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv2d(base_filter, base_filter // 2, 
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(base_filter // 2, 1, 
                               kernel_size=3, stride=1, padding=1, bias=True)
#         num_channels * (self.up ** 2)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
#         out = self.layers(x)
        out = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=True)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
#         out = self.pixel_shuffle(out)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
