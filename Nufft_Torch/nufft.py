from math import ceil
import Nufft_Torch.util as util
import Nufft_Torch.interp as interp
import numpy as np
import torch 
import Nufft_Torch.transforms as transforms


def nufft(input, coord, oversamp=1.25, width=4.0, device='cuda'):
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.clone()

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    # Zero-pad
    output = output / util.prod(input.shape[-ndim:]) ** 0.5
    output = util.resize(output, os_shape, device=device)

    # FFT
    output = transforms.fft2_cplx(output)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp, device)
    output = interp.interpolate(input=output, width=width, coord=coord, beta=beta, device=device)
    output = output/(width**ndim)

    return output



def nufft_adjoint(input, coord, out_shape, oversamp=1.25, width=4.0, device='cuda'):
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    out_shape = list(out_shape)

    os_shape = _get_oversamp_shape(out_shape, ndim, oversamp)

    # Gridding
    out_shape2 = out_shape.copy()
    os_shape2 = os_shape.copy()
    coord = _scale_coord(coord, out_shape2, oversamp, device)
    output = interp.gridding(input, os_shape2, width, coord, beta, device)
    # print('Fast NUFFTA output: ',output[0,0,245,245])
    output = output/(width**ndim)
    # IFFT
    output   = transforms.ifft2_cplx(output)
    # Crop
    output = util.resize(output, out_shape2, device=device)
    a = util.prod(os_shape2[-ndim:]) / util.prod(out_shape2[-ndim:]) ** 0.5
    output = output * a
    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    return output



def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [ceil(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [ceil(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i)
                                  for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, device):
    output = input.to(device)
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)
        # idx = torch.arange(i, dtype=output.dtype, device=device)
        idx = torch.arange(i, device=device)
        # Calculate apodization
        apod = (beta ** 2 - (np.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1)).to(device)

    return output