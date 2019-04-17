import torch
import torch.nn as nn
import numpy as np

from DCNSR.util import th_batch_map_offsets

class ConvOffset2D(nn.Conv2d):
    def __init__(self, filters, kernel_size, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self.ks = kernel_size
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, 2*self.ks*self.ks, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))
        
    def forward(self,x):
        # x_shape: (b,c,h,w)
        x_shape = x.size()
        
        # offsets: (b, 2*ks*ks, h, w)
        offsets = super(ConvOffset2D,self).forward(x)
        
        # offsets: (b, ks*ks, h, w, 2)
        offsets = self._to_b_ks_h_w_2(offsets, x_shape)
        
        # x_offsets: (b, c, h*ks, w*ks)
        x_offset, coords = th_batch_map_offsets(x, offsets, self.ks)
        
        return x_offset, coords
        
    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))
    
    @staticmethod
    def _to_b_ks_h_w_2(x, x_shape):
        """(b, 2*ks*ks, h, w) -> (b, ks*ks, h, w, 2)"""
        x = x.contiguous().view(int(x_shape[0]),-1, 2, int(x_shape[2]), int(x_shape[3]))
        x = x.contiguous().permute(0,1,3,4,2)
        return x