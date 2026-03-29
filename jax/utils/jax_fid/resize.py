import torch
from torch import Tensor

"""
Copied from torchscript_inception
"""

def forward(
    img: Tensor,
    ) -> Tensor:

  _0 = torch.nn.functional.affine_grid
  _1 = torch.nn.functional.grid_sample
  batch_size, channels, height, width, = img.shape
  x = img
  theta = torch.eye(2, 3, dtype=None, layout=None)
  _3 = torch.select(torch.select(theta, 0, 0), 0, 2)
  _4 = torch.select(torch.select(theta, 0, 0), 0, 0)
  _5 = torch.div(_4, width)
  _6 = torch.select(torch.select(theta, 0, 0), 0, 0)
  _7 = torch.add(_3, torch.sub(_5, torch.div(_6, 299)))
  _8 = torch.select(torch.select(theta, 0, 1), 0, 2)
  _9 = torch.select(torch.select(theta, 0, 1), 0, 1)
  _10 = torch.div(_9, height)
  _11 = torch.select(torch.select(theta, 0, 1), 0, 1)
  _12 = torch.add(_8, torch.sub(_10, torch.div(_11, 299)))
  _13 = torch.unsqueeze(theta, 0)
  theta0 = _13.repeat([batch_size, 1, 1])
  grid = _0(theta0, [batch_size, channels, 299, 299], False, )
  x0 = _1(x, grid, "bilinear", "border", False, )
  x1 = torch.sub(x0, 128)
  x2 = torch.div(x1, 128)

  return x2
