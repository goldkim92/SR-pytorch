
import os
import copy
import numpy as np
import scipy.misc as scm

import torch


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def th_to_b_c_hks_wks(vals, size, ks):
    """ (c, b, ks*ks, hw) -> (b, c, h*ks, w*ks)"""
    ks_y, ks_x = ks, ks
    vals = vals.permute(1,0,2,3) # (b, c, ks_y*ks_x, hw)
    vals = vals.contiguous().view(size[0], size[1], ks_y, ks_x, size[2], size[3]) # (b, c, ks_y, ks_x, h, w)
    vals = vals.unbind(dim=4)        # h * (b, c, ks_y, ks_x, w)
    vals = torch.cat(vals, dim=2)    # (b, c, h*ks_y, ks_x, w)
    vals = vals.unbind(dim=4)        # w * (b, c, h*ks_y, ks_x)
    vals = torch.cat(vals, dim=3)    # (b, c, h*ks_y, w*ks_x)
    return vals
    
    
def th_batch_map_coordinates(input, coords, ks=3):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, c, h, w)
    coords : tf.Tensor. shape = (b, ks*ks, hw, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """
    
    batch_size = input.size(0)
    channel_size = input.size(1)
    input_height = input.size(2)
    input_width = input.size(3)

    # coords.shape = (b, ks*ks, hw, 2)
    coords = torch.cat((torch.clamp(coords.narrow(3, 0, 1), 0, input_height - 1), 
                        torch.clamp(coords.narrow(3, 1, 1), 0, input_width - 1)), 3)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 3)
    coords_rt = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 3)
    idx = th_repeat(torch.arange(0, batch_size), ks * ks * input_height * input_width).long()
    idx.requires_grad = False
    idx = idx.to(device)

    def _get_vals_by_coords(input, coords):
        vals = input[idx, :, th_flatten(coords[...,0]), th_flatten(coords[...,1])] # (b*ks*ks*hw, c)
        vals = vals.contiguous().view(batch_size, ks*ks, input_height*input_width, -1) # (b, ks*ks, hw, c)
        vals = vals.permute(3,0,1,2) # (c, b, ks*ks, hw)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())
    
    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 1]*(vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 1]*(vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 0]* (vals_b - vals_t) + vals_t
    
    # make tensor size to (b, c, h*ks, w*ks)
    mapped_vals = th_to_b_c_hks_wks(mapped_vals, input.size(), ks)
    return mapped_vals, coords


def np_coordinate_set(h,w, is_kernel=False):
    """
    make [[0,0],[0,1],...,[0,w-1],...,[h-1,w-1]]
    with the shape : (hw, 2)
    if is_kernel is True,
    make [[-ks//2,-ks//2],...,[ks//2,ks//2]]
    """
    grid = np.meshgrid(range(h), range(w), indexing='ij')
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)
    if is_kernel is True:
        grid[:,0] = grid[:,0]-h//2
        grid[:,1] = grid[:,1]-w//2
    return grid
    

def th_generate_grid(batch_size, ks, input_height, input_width):
    kernel_grid = np_coordinate_set(ks, ks, is_kernel=True) # (ks*ks, 2)
    grid = np_coordinate_set(input_height, input_width, is_kernel=False) # (hw, 2)
    
    grid = np.stack([grid+kg for kg in kernel_grid], axis=0) # (ks*ks, hw, 2)
    grid = np.stack([grid]*batch_size, axis=0) # (b, ks*ks, hw, 2)
    grid = torch.from_numpy(grid).type(torch.float32)
    grid.requires_grad = False
    grid = grid.to(device)
    return grid


def th_batch_map_offsets(input, offsets, ks=3):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, c, h, w)
    offsets: torch.Tensor. shape = (b, ks*ks, h, w, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = input.size(0)
    input_height = input.size(2)
    input_width = input.size(3)

    # shape of offsets, grid, coord : (b, ks*ks, hw, 2)
    offsets = offsets.view(batch_size, ks*ks, -1, 2)
    grid = th_generate_grid(batch_size, ks, input_height, input_width)
    coords = offsets + grid

    mapped_vals, coords = th_batch_map_coordinates(input, coords, ks)
    return mapped_vals, coords