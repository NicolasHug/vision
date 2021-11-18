import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bilinear_sampler, coords_grid, upflow8


class ResidualBlock(nn.Module):
    # TODO: This is very similar to resnet.BasicBlock except for one call to relu
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super().__init__()

        # Note: bias=False because batchnorm would cancel the bias anyway
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), norm_layer(out_channels)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class FeatureEncoder(nn.Module):
    def __init__(self, out_channels=256, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, norm_layer=norm_layer, downsample=False)
        self.layer2 = self._make_layer(64, 96, norm_layer=norm_layer, downsample=True)
        self.layer3 = self._make_layer(96, 128, norm_layer=norm_layer, downsample=True)

        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, norm_layer, downsample):
        resblock1 = ResidualBlock(in_channels, out_channels, norm_layer=norm_layer, stride=(2 if downsample else 1))
        resblock2 = ResidualBlock(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(resblock1, resblock2)

    def forward(self, image1, image2=None):

        if image2 is not None:
            x = torch.cat([image1, image2], dim=0)
        else:
            x = image1

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        torch._assert(x.shape[-1] == image1.shape[-1] / 8, "The encoder should downsample H and W by 8")
        torch._assert(x.shape[-2] == image1.shape[-2] / 8, "The encoder should downsample H and W by 8")

        if image2 is not None:
            batch_size = image1.shape[0]
            x = torch.split(x, [batch_size, batch_size], dim=0)

        return x


class MotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super().__init__()

        corr_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(corr_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    # TODO :check core implem?
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
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


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, motion_encoder, hidden_dim=128):
        super().__init__()
        self.encoder = motion_encoder
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # # scale mask to balance gradients
        mask = self.mask(net)
        return net, mask, delta_flow


class CorrBlock:
    def __init__(self, num_levels=4, radius=4):
        self.radius = radius
        self.num_levels = num_levels
        self.corr_pyramid = []

    def build_pyramid(self, fmap1, fmap2):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        def compute_corr_volume(fmap1, fmap2):
            batch_size, num_channels, h, w = fmap1.shape
            fmap1 = fmap1.view(batch_size, num_channels, h * w)
            fmap2 = fmap2.view(batch_size, num_channels, h * w)

            corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
            corr = corr.view(batch_size, h, w, 1, h, w)
            return corr / torch.sqrt(torch.tensor(num_channels))

        torch._assert(fmap1.shape == fmap2.shape, "Input feature maps should have the same shapes")
        corr_volume = compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, num_channels, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
        self.corr_pyramid.append(corr_volume)
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        # The neighborhood of a centroid pixel x' is {x' + delta, ||delta||_inf < radius}
        # so it's a square surrounding x', with size 2 * radius + 1
        # The paper claims that it's ||.||_1 instead of ||.||_inf but the original code uses infinity-norm.
        neighborhood_size = 2 * self.radius + 1
        di = dj = torch.linspace(-self.radius, self.radius, neighborhood_size)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), axis=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_size, neighborhood_size, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, neigh_size, neigh_size, 2)
            indexed_corr_volume = bilinear_sampler(corr_volume, sampling_coords).view(batch_size, h, w, -1)
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        indexed_pyramid = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        torch._assert(
            indexed_pyramid.shape == (batch_size, self.num_levels * neighborhood_size ** 2, h, w),
            "Output shape of index pyramid is incorrect",
        )

        return indexed_pyramid


class RAFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_dim = 128
        self.context_dim = 128

        self.corr_radius = corr_levels = 4

        self.corr_block = CorrBlock(num_levels=4, radius=4)

        # feature network, context network, and update block
        self.fnet = FeatureEncoder(out_channels=256, norm_layer=nn.InstanceNorm2d)
        self.cnet = FeatureEncoder(out_channels=(self.hidden_dim + self.context_dim), norm_layer=nn.BatchNorm2d)
        motion_encoder = MotionEncoder(corr_levels=corr_levels, corr_radius=self.corr_radius)
        self.update_block = UpdateBlock(motion_encoder=motion_encoder, hidden_dim=self.hidden_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, num_flow_updates=12):
        """Estimate optical flow between pair of frames"""

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet(image1, image2)

        self.corr_block.build_pyramid(fmap1, fmap2)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        bs, _, h, w = image1.shape
        coords0 = coords_grid(bs, h // 8, w // 8).cuda()
        coords1 = coords_grid(bs, h // 8, w // 8).cuda()

        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()
            corr = self.corr_block.index_pyramid(coords1)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        return flow_predictions
