import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import CompactUNet, ConvBlock, edge_magnitude, gaussian_blur, group_count, time_features


class StructureEncoder(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = CompactUNet(3, features, width)

    def forward(self, ldct):
        return self.network(torch.cat((ldct, gaussian_blur(ldct), edge_magnitude(ldct)), dim=1))


class PartialEndpointPredictor(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.predictor = CompactUNet(1 + features + 3, 1, width, zero_output=True)

    def forward(self, ldct, tau, structure):
        times = time_features(ldct, tau)
        delta = self.predictor(torch.cat((ldct, structure, times), dim=1))
        return ldct + (1. - times[:, :1]) * delta


class ResidualVelocityField(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = CompactUNet(2 + features + 3, 1, width, zero_output=True)

    def forward(self, state, time, ldct, structure):
        return self.network(torch.cat((state, ldct, structure, time_features(state, time)), dim=1))


class GuidedVelocityUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, guide_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.guide = nn.Sequential(nn.Conv2d(guide_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(group_count(out_channels), out_channels), nn.SiLU())
        self.fusion = ConvBlock(2 * out_channels, out_channels)

    def forward(self, features, guide):
        features = self.upsample(features)
        if features.shape[-2:] != guide.shape[-2:]:
            raise ValueError("Guidance must match the learned upsampling scale")
        return self.fusion(torch.cat((features, self.guide(guide)), dim=1))


class StructureGuidedVelocitySuperResolver(nn.Module):
    def __init__(self, features, width, scale_factor):
        super().__init__()
        if scale_factor < 2 or scale_factor & (scale_factor - 1):
            raise ValueError("Scale factor must be a power of two greater than one")
        self.scale_factor = scale_factor
        stages = int(math.log2(scale_factor))
        guide_channels = 2 + features + 3
        channels = [max(16, width)]
        for _ in range(stages):
            channels.append(max(12, channels[-1] // 2))
        self.coarse_encoder = ConvBlock(1 + guide_channels, channels[0])
        self.stages = nn.ModuleList([GuidedVelocityUpsampleBlock(channels[i], channels[i + 1], guide_channels)
                                     for i in range(stages)])
        self.refinement = nn.Sequential(ConvBlock(channels[-1], channels[-1]), nn.Conv2d(channels[-1], 1, 3, padding=1))
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)
        self.velocity_skip = nn.ModuleList([nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
                                            for _ in range(stages)])
        coordinates = torch.arange(4, dtype=torch.float32)
        values = 1. - (coordinates - 1.5).abs() / 2.
        kernel = torch.outer(values, values)
        with torch.no_grad():
            for layer in self.velocity_skip:
                layer.weight[0, 0].copy_(kernel)

    def _guide(self, state, time, ldct, structure, size):
        values = [value if value.shape[-2:] == size else F.adaptive_avg_pool2d(value, size)
                  for value in (state, ldct, structure)]
        return torch.cat((*values, time_features(values[0], time)), dim=1)

    def forward(self, coarse, state, time, ldct, structure):
        expected = tuple(value * self.scale_factor for value in coarse.shape[-2:])
        if state.shape[-2:] != expected:
            raise ValueError("Coarse and restoration resolutions must match velocity_scale")
        coarse = coarse.to(state.dtype)
        features = self.coarse_encoder(torch.cat((coarse, self._guide(state, time, ldct, structure, coarse.shape[-2:])), dim=1))
        skip = coarse
        for stage, skip_stage in zip(self.stages, self.velocity_skip):
            size = tuple(value * 2 for value in features.shape[-2:])
            features = stage(features, self._guide(state, time, ldct, structure, size))
            skip = skip_stage(skip)
        return skip + self.refinement(features)


class LDCTAnchoredResidualRefinement(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = CompactUNet(3 + features, 2, width, zero_output=True)

    def forward(self, ldct, candidate, structure):
        displacement = candidate - ldct
        residual, logits = self.network(torch.cat((ldct, displacement, displacement.abs(), structure), dim=1)).chunk(2, dim=1)
        gate = torch.sigmoid(logits)
        return ldct + gate * residual, residual, gate


class IC_SABRFBackbone(nn.Module):
    def __init__(self, features, structure_width, endpoint_width, velocity_width,
                 residual_width, refinement_width, velocity_scale):
        super().__init__()
        dimensions = (features, structure_width, endpoint_width, velocity_width,
                      residual_width, refinement_width, velocity_scale)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("Network dimensions must be positive integers")
        self.structure_encoder = StructureEncoder(features, structure_width)
        self.endpoint_predictor = PartialEndpointPredictor(features, endpoint_width)
        self.velocity_superresolver = StructureGuidedVelocitySuperResolver(features, velocity_width, velocity_scale)
        self.residual_velocity = ResidualVelocityField(features, residual_width)
        self.output_refinement = LDCTAnchoredResidualRefinement(features, refinement_width)

    def forward(self, coarse_velocity, state, time, ldct, structure=None):
        if structure is None:
            structure = self.structure_encoder(ldct)
        guided = self.velocity_superresolver(coarse_velocity, state, time, ldct, structure)
        residual = self.residual_velocity(state, time, ldct, structure)
        return guided + residual
