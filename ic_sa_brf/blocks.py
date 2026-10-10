import math

import torch
from torch import nn
import torch.nn.functional as F


def time_vector(reference, time):
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError("Time must be scalar or have one value per image")
    return value


def time_features(reference, time):
    value = time_vector(reference, time)[:, None, None, None]
    value = value.expand(-1, -1, *reference.shape[-2:])
    return torch.cat((value, torch.sin(math.pi * value), torch.cos(math.pi * value)), dim=1)


def gaussian_blur(image):
    if min(image.shape[-2:]) < 5:
        raise ValueError("Spatial dimensions must be at least five pixels")
    coordinates = torch.arange(5, device=image.device, dtype=image.dtype) - 2
    kernel = torch.exp(-coordinates.square() / 2.)
    kernel = torch.outer(kernel, kernel) / kernel.sum().square()
    return F.conv2d(F.pad(image, (2,) * 4, mode="reflect"),
                    kernel.expand(image.shape[1], 1, 5, 5), groups=image.shape[1])


def signed_edges(image):
    padded = F.pad(image, (1,) * 4, mode="reflect")
    horizontal = padded[..., 2:] - padded[..., :-2]
    vertical = padded[..., 2:, :] - padded[..., :-2, :]
    gx = (horizontal[..., :-2, :] + 2. * horizontal[..., 1:-1, :] + horizontal[..., 2:, :]) / 8.
    gy = (vertical[..., :-2] + 2. * vertical[..., 1:-1] + vertical[..., 2:]) / 8.
    return gx, gy


def edge_magnitude(image):
    gx, gy = signed_edges(image)
    return torch.linalg.vector_norm(torch.stack((gx, gy), dim=0), dim=0)


def group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels), nn.SiLU())
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, value):
        return (self.block(value) + self.skip(value)) / math.sqrt(2.)


class CompactUNet(nn.Module):
    def __init__(self, in_channels, out_channels, width, zero_output=False):
        super().__init__()
        if width < 1:
            raise ValueError("Network width must be positive")
        self.enc0 = ConvBlock(in_channels, width)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.enc1 = ConvBlock(2 * width, 2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.mid = ConvBlock(4 * width, 4 * width)
        self.up1 = ConvBlock(6 * width, 2 * width)
        self.up0 = ConvBlock(3 * width, width)
        self.out = nn.Conv2d(width, out_channels, 3, padding=1)
        if zero_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, value):
        first = self.enc0(value)
        second = self.enc1(self.down1(first))
        middle = self.mid(self.down2(second))
        middle = F.interpolate(middle, size=second.shape[-2:], mode="bilinear", align_corners=False)
        middle = self.up1(torch.cat((middle, second), dim=1))
        middle = F.interpolate(middle, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.up0(torch.cat((middle, first), dim=1)))
