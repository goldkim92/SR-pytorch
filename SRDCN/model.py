import torch
import torch.nn as nn

import copy
from DCNSR.deform_conv import ConvOffset2D

class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv_offset2 = ConvOffset2D(base_filter, 3)
        self.conv2 = nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=3, stride=3, padding=0, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv_offset3 = ConvOffset2D(base_filter//2, 3)
        self.conv3 = nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=3, stride=3, padding=0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        
    def forward(self, x):
#         out = self.layers(x)
        out = self.relu1(self.conv1(x))
    
        out, self.coords2 = self.conv_offset2(out)
        out = self.relu2(self.conv2(out))
        
        out, self.coords3 = self.conv_offset3(out)
        out = self.conv3(out)
        
        out = self.pixel_shuffle(out)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def freeze(self, module_classes):
        '''Freeze modules for finetuning'''
        for k,m in self._modules.items():
            if any([type(m)==mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
        
def get_deform_mil(trainable=True, num_channels=1, base_filter=64, upscale_factor=3, freeze_filter=[nn.Conv2d, nn.Linear]):
    model = Net(num_channels, base_filter, upscale_factor)
    if not trainable:
        model.freeze(freeze_filter)
    return model


def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys():
        if not k in wf:
            wf[k] = wt[k]
    model_to.load_state_dict(wf)