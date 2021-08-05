from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import bilinear_sampler, coords_grid, upflow8, set_attributes


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        dim_inner: int,
        dim_out: int,
        stride: int = 1,
        conv: Callable = nn.Conv2d,
        norm: Callable = nn.BatchNorm2d,
        act: Callable = nn.ReLU,
    ) -> None:
        super(BottleneckBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            conv(dim_in, dim_inner, kernel_size=1),
            norm(dim_inner),
            act(),
            conv(dim_inner, dim_inner, kernel_size=3, padding=1, stride=stride),
            norm(dim_inner),
            act(),
            conv(dim_inner, dim_out, kernel_size=1),
            norm(dim_out),
            act(),
        )
        self.act = act()
        if stride > 1:
            self.downsample = nn.Sequential(
                conv(dim_in, dim_out, kernel_size=1, stride=stride),
                norm(dim_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "downsample"):
            out = self.act(self.downsample(x) + self.bottleneck(x))
        else:
            out = self.act(x + self.bottleneck(x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int = 3,
        dim_out: int = 128,
        dropout_rate: float = 0.0,
        conv: Callable = nn.Conv2d,
        norm: Callable = nn.BatchNorm2d,
        block: Callable = BottleneckBlock,
        act: Callable = nn.ReLU,
    ) -> None:

        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(dim_in, 32, kernel_size=7, stride=2, padding=3),
            norm(32),
            act(),
            block(dim_in=32, dim_inner=8, dim_out=32, stride=1),
            block(dim_in=32, dim_inner=8, dim_out=64, stride=2),
            block(dim_in=64, dim_inner=8, dim_out=96, stride=2),
            conv(96, dim_out, kernel_size=1),
        )
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this design is nasty, don't do tuple
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.encoder(x)

        if self.training and hasattr(self, "dropout"):
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class BasicMotionEncoder(nn.Module):
    def __init__(
        self,
        *,
        corr_radius: int,
        corr_levels: int,
    ) -> None:
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(
        self,
        flow: torch.Tensor,
        corr: torch.Tensor,
    ) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        input_dim: int = 192 + 128,
        conv: Callable = nn.Conv2d,
    ) -> None:
        super(SepConvGRU, self).__init__()
        self.convz1 = conv(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = conv(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = conv(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = conv(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = conv(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = conv(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        *,
        corr_radius: int,
        corr_levels: int,
        hidden_dim: int = 128,
        input_dim: int = 128,
        conv: Callable = nn.Conv2d,
        act: Callable = nn.ReLU,
    ) -> None:
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(
            corr_radius=corr_radius,
            corr_levels=corr_levels,
        )
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim,
            input_dim=128 + hidden_dim,
        )
        self.flow_head = nn.Sequential(
            conv(input_dim, 256, 3, padding=1), act(), conv(256, 2, 3, padding=1)
        )
        self.mask = nn.Sequential(
            conv(128, 256, 3, padding=1),
            act(inplace=True),
            conv(256, 64 * 9, 1, padding=0),
        )

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
        upsample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


# TODO: optimize the code logic
class CorrBlock:
    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ) -> None:
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class RAFT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        super(RAFT, self).__init__()
        set_attributes(self, locals())
        self.fnet = Encoder(
            dim_out=256,
            norm=nn.InstanceNorm2d,
            dropout_rate=dropout_rate,
        )
        self.cnet = Encoder(
            dim_out=hidden_dim + context_dim,
            norm=nn.BatchNorm2d,
            dropout_rate=dropout_rate,
        )
        self.update_block = BasicUpdateBlock(
            corr_radius=corr_radius,
            corr_levels=corr_levels,
            hidden_dim=hidden_dim,
        )

    def initialize_flow(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        flow_init: bool = None,
        upsample: bool = True,
        test_mode: bool = False,
    ) -> torch.Tensor:
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        print(image1.shape, image2.shape)

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        print('in corrblock')
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
        print('out of corrbloack')
        print(fmap1.shape)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            print("update itr", itr)
            coords1 = coords1.detach()
            print("before corr_fn")
            corr = corr_fn(coords1)  # index correlation volume
            print("after corr_fn")

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
