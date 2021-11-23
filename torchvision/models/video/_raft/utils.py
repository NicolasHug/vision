import torch
import torch.nn.functional as F


def grid_sample(img, absolute_grid, *args, **kwargs):
    """Same as torch's grid_sample, with absolute pixel coordinates instead of normalized coordinates."""
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = torch.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, *args, **kwargs)


def make_coords_grid(batch_size, h, w):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)
