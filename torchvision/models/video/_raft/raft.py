import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bilinear_sampler, coords_grid, upflow8


class ResidualBlock(nn.Module):
    # TODO: This is very similar to resnet.BasicBlock except for one call to relu
    # TODO: remove call to relu gives really bad results - is this because of initialization?
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
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        return x


class MotionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convc1 = nn.Conv2d(in_channels, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64 + 192, out_channels - 2, 3, padding=1)  # -2 because we cat the flow

    def forward(self, flow, corr_features):
        cor = F.relu(self.convc1(corr_features))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, *, input_size, hidden_size):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_size + input_size, hidden_size, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_size + input_size, hidden_size, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_size + input_size, hidden_size, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_size + input_size, hidden_size, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_size + input_size, hidden_size, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_size + input_size, hidden_size, (5, 1), padding=(2, 0))

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
    def __init__(self, *, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, *, motion_encoder, reccurrent_block, flow_head):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.reccurrent_block = reccurrent_block
        self.flow_head = flow_head

    def forward(self, hidden_state, context, corr_features, flow):
        motion_features = self.motion_encoder(flow, corr_features)
        x = torch.cat([context, motion_features], dim=1)

        hidden_state = self.reccurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return hidden_state, delta_flow


class CorrBlock:
    def __init__(self, *, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

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
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        """Return correlation features by indexing from the pyramid."""
        # The neighborhood of a centroid pixel x' is {x' + delta, ||delta||_inf <= radius}
        # so it's a square surrounding x', with size 2 * radius + 1
        # The paper claims that it's ||.||_1 instead of ||.||_inf but the original code uses infinity-norm.
        neighborhood_size = 2 * self.radius + 1
        # TODO: benchmark to figure out whether we should make it cuda from the start
        di = torch.linspace(-self.radius, self.radius, neighborhood_size)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_size)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), axis=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_size, neighborhood_size, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, neigh_size, neigh_size, 2)
            # TODO: Could this be optimized a bit?
            indexed_corr_volume = bilinear_sampler(corr_volume, sampling_coords).view(batch_size, h, w, -1)
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

        torch._assert(
            corr_features.shape == (batch_size, self.num_levels * neighborhood_size ** 2, h, w),
            "Output shape of index pyramid is incorrect",
        )

        return corr_features


def raft():

    num_channels_encoder = 256
    feature_encoder = FeatureEncoder(out_channels=num_channels_encoder, norm_layer=nn.InstanceNorm2d)
    context_encoder = FeatureEncoder(out_channels=num_channels_encoder, norm_layer=nn.BatchNorm2d)

    num_levels = 4
    radius = 4
    corr_block = CorrBlock(num_levels=num_levels, radius=radius)

    # TODO: a bit sad we have to re-compute this
    # Make this out_channels attribute
    corr_block_out_channels = num_levels * (2 * radius + 1) ** 2  # see comments in index_pyramid()
    motion_encoder_out = 128
    motion_encoder = MotionEncoder(in_channels=corr_block_out_channels, out_channels=motion_encoder_out)

    hidden_state_size = context_size = num_channels_encoder // 2
    gru = SepConvGRU(input_size=motion_encoder_out + context_size, hidden_size=hidden_state_size)

    flow_head = FlowHead(in_channels=hidden_state_size, hidden_size=256)

    update_block = UpdateBlock(motion_encoder=motion_encoder, reccurrent_block=gru, flow_head=flow_head)

    mask_predictor = nn.Sequential(
        nn.Conv2d(hidden_state_size, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        # 8 * 8 * 9 because the predicted flow is downsampled by 8, from the downsampling of the initial FeatureEncoder
        # and we interpolate with all 9 surrounding neighbors. See paper and appendix
        nn.Conv2d(256, 8 * 8 * 9, 1, padding=0),
    )

    return RAFT(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor,
    )


class RAFT(nn.Module):
    def __init__(self, *, feature_encoder, context_encoder, corr_block, update_block, mask_predictor):
        super().__init__()

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block

        # TODO Should this be called mask_head?
        # TODO Should it be part of the flow updater?
        self.mask_predictor = mask_predictor  

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _upsample_flow(self, flow, up_mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        up_mask = up_mask.view(N, 1, 9, 8, 8, H, W)
        up_mask = torch.softmax(up_mask, dim=2)

        upsampled_flow = F.unfold(8 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)

        upsampled_flow = torch.sum(up_mask * upsampled_flow, dim=2)
        upsampled_flow = upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(N, 2, 8 * H, 8 * W)
        return upsampled_flow

    def forward(self, image1, image2, num_flow_updates=12):

        batch_size, _, h, w = image1.shape
        torch._assert((h % 8 == 0) and (w % 8 == 0), "input image H and W should be divisible by 8")

        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.split(fmaps, (batch_size, batch_size), dim=0)
        torch._assert(fmap1.shape[-2:] == (h / 8, w / 8), "The encoder should downsample H and W by 8")

        self.corr_block.build_pyramid(fmap1, fmap2)

        # TODO: should these be different blocks??
        context_out = self.context_encoder(image1)
        # TODO use chunk
        hidden_state, context = torch.split(context_out, (context_out.shape[1] // 2, context_out.shape[1] // 2), dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = torch.relu(context)

        coords0 = coords_grid(batch_size, h // 8, w // 8).cuda()
        coords1 = coords_grid(batch_size, h // 8, w // 8).cuda()
        flow = coords1 - coords0  # initial flow is zero everywhere

        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)

            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            up_mask = self.mask_predictor(hidden_state)
            upsampled_flow = self._upsample_flow(flow=flow, up_mask=up_mask)
            flow_predictions.append(upsampled_flow)

        return flow_predictions
