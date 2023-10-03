
import torch
import numpy
import Nufft_Torch.util as util
#test


def kb_op(x, beta):
    # if abs(x) > 1:
    #     return 0
    # print('x1:', (1-x**2)**0.5)
    zeros_mask = (abs(x)<1).type(torch.int)
    x = x*zeros_mask
    # print(x)
    x = beta * (1 - x**2)**0.5
    # print('x:', x)
    t = x / 3.75
    mask1 = (x<3.75)
    mask2 = ~mask1
    # print('x:', x)
    mask1 = mask1.type(torch.int)
    mask2 = mask2.type(torch.int)
    # print(mask1, mask2)
    return  zeros_mask*(mask1*(1 + 3.5156229 * t**2 + 3.0899424 * t**4 + 1.2067492 * t**6\
         + 0.2659732 * t**8 +0.0360768 * t**10 + 0.0045813 * t**12)\
         + mask2*(x**-0.5 * torch.exp(x) * (0.39894228 + 0.01328592 \
            * t**-1 + 0.00225319 * t**-2 - 0.00157565 * t**-3 + 0.00916281 * t**-4\
             - 0.02057706 * t**-5 + 0.02635537 * t**-6 - 0.01647633 * t**-7 + 0.00392377 * t**-8)))



def interpolate(input, width, coord, beta, device):
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size, npts], dtype=input.dtype, device=device)

    output = _interpolate2(output, input, width, coord, beta)

    return output.reshape(batch_shape + pts_shape)


def _interpolate2(output, input, width, coord, beta):
    batch_size, ny, nx = input.shape

    kx, ky = coord[:, -1], coord[:, -2]

    for dy in range(int(-width/2),int(width/2)):
        dy_list = torch.ceil(ky + dy)
        wy = kb_op((ky-dy_list) / (width / 2), beta)

        for dx in range(int(-width/2),int(width/2)):
            dx_list = torch.ceil(kx + dx)
            w = wy * kb_op((kx - dx_list) / (width / 2), beta)

            output = output+ w * input[:, dy_list.type(torch.long)%ny, dx_list.type(torch.long)%nx]

    return output




def gridding(input, shape, width, coord, beta, device):
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype, device=device)
    # print('output shape:', output.shape)
    output=_gridding2(output, input, width, beta, coord)

    return output.reshape(shape)


def _gridding2(output, input, width, beta, coord):
    batch_size, ny, nx = output.shape

    kx, ky = coord[:,-1], coord[:,-2]

    for dy in range(int(-width/2),int(width/2)):
        dy_list = torch.ceil(ky + dy).type(torch.long)
        wy = kb_op((dy_list-ky)/(width/2), beta)

        for dx in range(int(-width/2), int(width/2)):

            dx_list = torch.ceil(kx + dx).type(torch.long)
            w = wy * kb_op((dx_list-kx)/(width/2), beta)


            for b in range(batch_size):
                update = torch.zeros_like(output)
                update.index_put_((torch.tensor(b).type(torch.long), dy_list%ny, dx_list%nx), w*input[b], accumulate=True)
                output = output + update
                
    return output
